#include <HYPRE_struct_ls.h>
#include <spark/constants/constants.h>
#include "HYPRE_utilities.h"
#include "_hypre_utilities.h"
#include "log/log.h"
#include "spark/core/matrix.h"
#include "spark/core/vec.h"
#include "spark/em/poisson.h"

using namespace spark::core;
using namespace spark::em;
using namespace spark::spatial;

namespace {
    int stencil_indices[5] = {0, 1, 2, 3, 4};
    int opposite_indices[] = {0, 2, 1, 4, 3};
    int stencil_offsets[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    constexpr double solver_tolerance = 1e-6;
    bool hypre_initialized = false;
}

CylindricalPoissonSolver2D::~CylindricalPoissonSolver2D() = default;

struct CylindricalPoissonSolver2D::Impl {
    HYPRE_StructGrid hypre_grid_ = nullptr;
    HYPRE_StructStencil hypre_stencil_ = nullptr;
    HYPRE_StructMatrix hypre_A_ = nullptr;
    HYPRE_StructVector hypre_b_ = nullptr;
    HYPRE_StructVector hypre_x_ = nullptr;
    HYPRE_StructSolver hypre_solver_ = nullptr;

    struct BoundaryRef {
        IntVec<2> pos;
        IntVec<2> offset;
        Region* boundary = nullptr;
    };

    struct CellProp {
        CellType cell_type = CellType::Internal;
        Region* region = nullptr;
    };

    TMatrix<CellProp, 2> cells_;
    std::vector<Region> boundaries_;

    DomainProp prop_;

    void assemble();
    void solve(Matrix<2>& out, const Matrix<2>& rho);

    Impl(const DomainProp& prop, const std::vector<Region>& boundaries)
            : prop_(prop), boundaries_(boundaries) {
            if (!hypre_initialized) {
                HYPRE_Initialize();
                hypre_initialized = true;
            }
            input_cache_.resize(prop_.extents.to<size_t>());
        }
    ~Impl();

private:
    void create_grid();
    void create_matrices();
    void create_stencil();

    void set_stencils();
    void set_cells();

    CellType get_cell(int i, int j);
    core::Matrix<2> input_cache_;
    std::vector<BoundaryRef> boundary_refs_;
};

void CylindricalPoissonSolver2D::Impl::create_grid() {
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &hypre_grid_);
    int lower[] = {0, 0};
    int upper[] = {prop_.extents.x - 1, prop_.extents.y - 1};
    HYPRE_StructGridSetExtents(hypre_grid_, lower, upper);
    HYPRE_StructGridAssemble(hypre_grid_);
}

void CylindricalPoissonSolver2D::Impl::create_matrices() {
    HYPRE_StructStencilCreate(2, 5, &hypre_stencil_);
    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, hypre_grid_, hypre_stencil_, &hypre_A_);
    HYPRE_StructMatrixInitialize(hypre_A_);

    HYPRE_StructVectorCreate(MPI_COMM_WORLD, hypre_grid_, &hypre_b_);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, hypre_grid_, &hypre_x_);
    HYPRE_StructVectorInitialize(hypre_b_);
    HYPRE_StructVectorInitialize(hypre_x_);
}

void CylindricalPoissonSolver2D::Impl::create_stencil() {
    for (int n = 0; n < 5; n++)
        HYPRE_StructStencilSetElement(hypre_stencil_, n, &stencil_offsets[n][0]);
}

void CylindricalPoissonSolver2D::Impl::set_cells() {
    cells_.resize(ULongVec<2>(prop_.extents.x, prop_.extents.y));
    cells_.fill({});

    for (const auto& b : boundaries_) {
        cells_.fill({b.region_type, const_cast<Region*>(&b)}, b.lower_left, b.upper_right);
    }
}

void CylindricalPoissonSolver2D::Impl::set_stencils() {
    const auto [sz, sr] = cells_.size();
    const auto [dz, dr] = prop_.dx;

    const double idz2 = 1.0 / (dz * dz);
    const double idr2 = 1.0 / (dr * dr);

    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sr; ++j) {
            int index[] = {i, j};
            
            auto cell = cells_(i, j).cell_type;

            double coeff_center = 0.0;
            double coeff_left = 0.0;
            double coeff_right = 0.0;
            double coeff_down = 0.0;
            double coeff_up = 0.0;

            if (j == 0) {
                coeff_center = -4.0 * idr2 - 2.0 * idz2;
                coeff_left = idz2;
                coeff_right = idz2;
                coeff_down = 0.0;
                coeff_up = 4.0 * idr2;
            } else {
                double rj = static_cast<double>(j) * dr;

                coeff_left = idz2;
                coeff_right = idz2;
                coeff_down = idr2 - 0.5 / (rj * dr);
                coeff_up = idr2 + 0.5 / (rj * dr);
                coeff_center = - (coeff_left + coeff_right + coeff_down + coeff_up);
            }

            double stencil[5] = {coeff_center, coeff_left, coeff_right, coeff_down, coeff_up};

            if (cell == CellType::BoundaryDirichlet) {
                double stencil_dirichlet[] = {1.0};
                HYPRE_StructMatrixSetValues(hypre_A_, index, 1, stencil_indices, stencil_dirichlet);
                continue;
            }

            for (int p = 1; p < 5; ++p) {
                const int neighbor_pos[] = {i + stencil_offsets[p][0], j + stencil_offsets[p][1]};
                const auto neighbor_type = get_cell(neighbor_pos[0], neighbor_pos[1]);

                if (neighbor_type == CellType::BoundaryDirichlet) {
                    stencil[p] = 0.0;
                    boundary_refs_.push_back({{i, j},
                                              {stencil_offsets[p][0], stencil_offsets[p][1]},
                                              cells_(neighbor_pos[0], neighbor_pos[1]).region});
                }

                if (cell == CellType::BoundaryNeumann && neighbor_type == CellType::External) {
                    stencil[p] = 0.0;
                }

                if (cell == CellType::Internal && neighbor_type == CellType::External) {
                    SPARK_LOG_WARN("internal node [%d, %d] at boundary with offset [%d, %d]!", i, j,
                                   stencil_offsets[p][0], stencil_offsets[p][1]);
                }
            }

            HYPRE_StructMatrixSetValues(hypre_A_, index, 5, stencil_indices, stencil);
        }
    }
}

CellType CylindricalPoissonSolver2D::Impl::get_cell(int i, int j) {
    if (i >= 0 && i < prop_.extents.x && j >= 0 && j < prop_.extents.y) {
        return cells_(i, j).cell_type;
    }

    return CellType::External;
}

void CylindricalPoissonSolver2D::Impl::assemble() {
    create_grid();
    create_matrices();
    create_stencil();
    set_stencils();
    HYPRE_StructMatrixAssemble(hypre_A_);
}

void CylindricalPoissonSolver2D::Impl::solve(core::Matrix<2>& out, const core::Matrix<2>& rho) {
    constexpr double k = -1.0 / constants::eps0;
    for (int i = 0; i < rho.size().x; i++) {
        for (int j = 0; j < rho.size().y; ++j) {
            int pos[] = {i, j};
            HYPRE_StructVectorSetValues(hypre_b_, pos, rho(i, j) * k);
        }
    }

    for (const auto& region : boundaries_) {
        if (region.region_type == CellType::BoundaryDirichlet && region.input) {
            core::Matrix<2> cache;
            cache.resize({static_cast<size_t>(region.upper_right.x - region.lower_left.x + 1),
                          static_cast<size_t>(region.upper_right.y - region.lower_left.y + 1)});
            cache.fill(region.input());
            int ilower[] = {region.lower_left.x, region.lower_left.y};
            int iupper[] = {region.upper_right.x, region.upper_right.y};
            HYPRE_StructVectorSetBoxValues(hypre_b_, ilower, iupper, cache.data_ptr());
        }
    }

    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &hypre_solver_);
    HYPRE_StructSMGSetTol(hypre_solver_, solver_tolerance);
    HYPRE_StructSMGSetup(hypre_solver_, hypre_A_, hypre_b_, hypre_x_);
    HYPRE_StructSMGSolve(hypre_solver_, hypre_A_, hypre_b_, hypre_x_);

    out.resize(prop_.extents.to<size_t>());
    for (int i = 0; i < prop_.extents.x; ++i) {
        int idx1[] = {i, 0};
        int idx2[] = {i, prop_.extents.y - 1};
        HYPRE_StructVectorGetBoxValues(hypre_x_, idx1, idx2, &out(i, 0));
    }
    HYPRE_StructSMGDestroy(hypre_solver_);
}

CylindricalPoissonSolver2D::Impl::~Impl() {
    if (hypre_grid_)
        HYPRE_StructGridDestroy(hypre_grid_);
    if (hypre_stencil_)
        HYPRE_StructStencilDestroy(hypre_stencil_);
    if (hypre_A_)
        HYPRE_StructMatrixDestroy(hypre_A_);
    if (hypre_b_)
        HYPRE_StructVectorDestroy(hypre_b_);
    if (hypre_x_)
        HYPRE_StructVectorDestroy(hypre_x_);
}

CylindricalPoissonSolver2D::CylindricalPoissonSolver2D() : impl_(nullptr) {}

CylindricalPoissonSolver2D::CylindricalPoissonSolver2D(const DomainProp& prop,
                                                       const std::vector<Region>& regions)
    : impl_(std::make_unique<Impl>(prop, regions)) {
    impl_->assemble();
}
CylindricalPoissonSolver2D::CylindricalPoissonSolver2D(CylindricalPoissonSolver2D&& other) noexcept
    : impl_(std::move(other.impl_)) {}

CylindricalPoissonSolver2D& CylindricalPoissonSolver2D::operator=(CylindricalPoissonSolver2D&& other) noexcept {
    impl_ = std::move(other.impl_);
    return *this;
}
void CylindricalPoissonSolver2D::solve(core::Matrix<2>& out, const core::Matrix<2>& rho) {
    impl_->solve(out, rho);
}