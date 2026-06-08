#ifndef ABSTRACT_OPERATOR_H
#define ABSTRACT_OPERATOR_H

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "geom_helpers.hpp"
#include "mesh.hpp"
#include "mfem.hpp"

enum class MassLumping
{
    None,      ///< No lumping (full Galerkin mass matrix, cannot do DEC, expect
               ///< failure).
    RowSum,    ///< Take the absolute-value-row-sum of the (non-mass-lumped) Galerkin mass
               ///< matrix.
    ScaledId,  ///< Take h^{n - 2k} Id as the ``approximation'' of the mass
               ///< matrix for k-forms in n dimensions
    Barycentric  ///< Barycentric (geometric) mass lumping.
};

enum class BCond
{
    Essential,
    Natural
};

enum class OperatorMode
{
    Galerkin,  ///< Standard Finite Element Method (FEM).
    DEC        ///< Discrete Exterior Calculus (DEC).
};

// =====================================================================
//                         CLASS DECLARATION
// =====================================================================

/**
 * @class WhitneyFormOperator
 * @brief Unified base for operators using differential forms.
 * * Provides fine-grained Boundary Condition (BC) control and supports both
 * Finite Element Method (FEM) and Discrete Exterior Calculus (DEC, i.e.
 * mass-lumped) modes.
 *
 * @tparam DerivedOperator The specific operator implementation (CRTP).
 * @tparam ACTIVE_FORMS Variadic list of active form degrees (e.g., 0, 1, 2).
 */
template <class DerivedOperator, unsigned int... ACTIVE_FORMS>
class WhitneyFormOperator : public mfem::Operator
{
public:
    static constexpr int MAX_DIM = 3;  ///< Maximum spatial dimension supported.

protected:
    std::shared_ptr<Mesh> mesh_ptr_;  ///< Underlying computational mesh.
    mutable OperatorMode  opmode_;    ///< Current operation mode (FEM/DEC).
    MassLumping           ml_;        ///< Selected mass lumping scheme.
    const BCond bc_ = BCond::Natural;

    mfem::Array<int>             block_offsets_;
    std::array<int, MAX_DIM + 1> form_to_block_;

    std::array<std::shared_ptr<mfem::FiniteElementCollection>, MAX_DIM + 1>
                                                                       fecs_;
    std::array<std::shared_ptr<mfem::FiniteElementSpace>, MAX_DIM + 1> spaces_;
    std::array<std::shared_ptr<mfem::BilinearForm>, MAX_DIM + 1> mass_forms_;

    std::array<std::shared_ptr<mfem::Array<int>>, MAX_DIM + 1> essential_dof_;
    std::array<std::shared_ptr<mfem::Array<int>>, MAX_DIM + 1>
        essential_dof_masks_;

    std::array<mfem::Vector, MAX_DIM + 1> lumped_mass_vecs_;

    std::array<std::unique_ptr<mfem::SparseMatrix>, MAX_DIM>     d_;
    std::array<std::unique_ptr<mfem::SparseMatrix>, MAX_DIM + 1> delta_;

    // Orthogonalization data for the nullspace (trivial harmonics)
    int          harmonic_form_degree_ = 0;
    mfem::Vector c_;          // The constant function f(x)=1 (projected)
    mfem::Vector Mc_;         // The mass-weighted constant: M * c
    mfem::real_t cMc_ = 0.0;  // Inner product (c, c)_M = c^T * M * c

public:
    /**
     * @brief Constructor for WhitneyFormOperator.
     * @param mesh Shared pointer to the geometric mesh.
     * @param bc BCond::Essential or Bcond::Natural boundary conditions. Default
     * is natural.
     * @param mode Operating mode (Galerkin or DEC). Default is Galerkin.
     * @param ml Mass lumping strategy. Default is RowSum.
     */
    WhitneyFormOperator(
        std::shared_ptr<Mesh> mesh,
        const BCond           bc     = BCond::Natural,
        const OperatorMode    mode   = OperatorMode::Galerkin,
        const MassLumping     ml     = MassLumping::RowSum);

    virtual ~WhitneyFormOperator() = default;

    // =====================================================================
    //                   CORE MATH & INTERFACE OPERATIONS
    // =====================================================================

    /**
     * @brief Applies the operator to vector x, returning result in y.
     * @param x Input vector.
     * @param y Output vector.
     */
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /**
     * @brief Applies the Hodge star (mass matrix) operation to a BlockVector.
     * @param u Vector to which the Hodge star is applied.
     */
    void applyHodgeStar(mfem::BlockVector& u) const;

    /**
     * @brief Applies the inverse Hodge star operation.
     * @param u Vector to which the inverse Hodge star is applied.
     */
    void applyInvHodgeStar(mfem::BlockVector& u) const;

    /**
     * @brief Convenience interface for eliminateTrivialSolutionHarmonicsBlock
     */
    void eliminateTrivialSolutionHarmonics(mfem::Vector& u) const;

    /**
     * @brief Eliminates trivial solution harmonics (constants) from the given
     * vector.
     * @param u Vector to process.
     */
    void eliminateTrivialSolutionHarmonicsBlock(mfem::BlockVector& u) const;

    /**
     * @brief Eliminates the boundary unkowns (if present).
     * NOTE:  Will throw errors when called with natural bc.
     * @param x MFEM vector (or blockvector <- vector) to zero out
     */
    void eliminateBC(mfem::Vector& x) const;

    // =====================================================================
    //                   BLOCK MANAGEMENT & ACCESSORS
    // =====================================================================

    /**
     * @brief Creates an appropriately sized BlockVector for this operator.
     * @return A newly allocated BlockVector.
     */
    [[nodiscard]] mfem::BlockVector createBlockVector() const;

    /**
     * @brief Gets the block offsets array.
     * @return Const reference to the block offsets.
     */
    [[nodiscard]] const mfem::Array<int>& getBlockOffsets() const
    {
        return block_offsets_;
    }

    /**
     * @brief Gets the size of a specific form block.
     * @param k The form degree.
     * @return Size of the block.
     */
    [[nodiscard]] unsigned int getBlockSize(const unsigned int k) const;

    /**
     * @brief Extracts a specific form block from a full BlockVector.
     * @param bv The source BlockVector.
     * @param k The form degree to extract.
     * @return Reference to the specific block.
     */
    mfem::Vector& getFormBlock(mfem::BlockVector& bv, const unsigned int k)
        const;

    // --- Space and Topology Getters ---

    [[nodiscard]] std::shared_ptr<const mfem::Array<int>> getEssentialTrueDofs(
        const unsigned int k) const
    {
        return (k <= MAX_DIM) ? essential_dof_[k] : nullptr;
    }

    [[nodiscard]] std::shared_ptr<mfem::FiniteElementSpace> getSpace(
        const unsigned int k)
    {
        return (k <= MAX_DIM) ? spaces_[k] : nullptr;
    }

    [[nodiscard]] std::shared_ptr<const mfem::FiniteElementSpace> getSpace(
        const unsigned int k) const
    {
        return (k <= MAX_DIM) ? spaces_[k] : nullptr;
    }

    [[nodiscard]] std::shared_ptr<mfem::BilinearForm> getMassMatrix(
        const unsigned int k)
    {
        return (k <= MAX_DIM) ? mass_forms_[k] : nullptr;
    }

    [[nodiscard]] std::shared_ptr<const mfem::BilinearForm> getMassMatrix(
        const unsigned int k) const
    {
        return (k <= MAX_DIM) ? mass_forms_[k] : nullptr;
    }

    [[nodiscard]] const mfem::SparseMatrix& getExteriorDerivative(
        const unsigned int k) const
    {
        return *d_.at(k);
    }

    [[nodiscard]] const mfem::SparseMatrix& getCodifferential(
        const unsigned int k) const
    {
        return *delta_.at(k);
    }

    [[nodiscard]] const mfem::Vector& getLumpedMass(const unsigned int k) const
    {
        return lumped_mass_vecs_.at(k);
    }

    // =====================================================================
    //                   CONFIGURATION GETTERS & SETTERS
    // =====================================================================

    [[nodiscard]] std::shared_ptr<const Mesh> getMesh() const
    {
        return mesh_ptr_;
    }

    /**
     * @brief Dynamically switches the operator mode (e.g., FEM to DEC).
     * @param mode The new OperatorMode to set.
     */
    void setMode(const OperatorMode mode)
    {
        opmode_ = mode;
        setupTrivialHarmonics();
    }

    [[nodiscard]] OperatorMode getMode() const { return opmode_; }
    [[nodiscard]] MassLumping  getMassLumping() const { return ml_; }
    [[nodiscard]] BCond        getBCond() const { return bc_; }

    // =====================================================================
    //                   ASSEMBLY AND VIRTUAL INTERFACE
    // =====================================================================

    /**
     * @brief Generates a random vector compatible with the operator's domain.
     * @param seed Random seed. Default 0 (time, non-deterministic (?), mfem
     * stuff).
     * @return The generated random BlockVector.
     */
    [[nodiscard]] mfem::BlockVector getRandomVector(int seed = 0) const;

    /**
     * @brief Assembles the operator matrix based on the current mode.
     * @return Shared pointer to the assembled SparseMatrix.
     */
    [[nodiscard]] std::shared_ptr<mfem::SparseMatrix> assembleMatrix()
    {
        return (opmode_ == OperatorMode::Galerkin) ? assembleFem()
                                                   : assembleDec();
    }

    // --- Pure Virtual Implementation Hooks ---
    virtual void applyDecOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const = 0;
    virtual void applyFemOp(const mfem::BlockVector& in, mfem::BlockVector& out)
        const = 0;

    virtual std::shared_ptr<mfem::SparseMatrix> assembleDec() = 0;
    virtual std::shared_ptr<mfem::SparseMatrix> assembleFem()
    {
        return nullptr;
    }

    /**
     * @brief Creates a refined version of this operator on a refined mesh.
     * @return Shared pointer to the operator on the (uniformly) refined mesh.
     */
    std::shared_ptr<DerivedOperator> createRefined() const;

protected:
    void setupBlockStructure();
    void assembleSpace(const unsigned int k);
    void assembleExteriorDerivative(const unsigned int k);
    void assembleGalerkinMass(const unsigned int k);
    void buildEssentialDofMask(const int k);
    void assembleGeometricLumpedMasses();
    void assembleGeometric2D(const DualMesh dual);
    void assembleGeometric3D(const DualMesh dual);
    void computeDecOperators(const unsigned int k);
    void eliminateBoundaryConditionsD(const unsigned int k);
    void eliminateBoundaryConditionsDelta(const unsigned int k);
    void eliminateBoundaryConditionsMass(const unsigned int k);
    void eliminateAllBoundaryConditions();
    void setupTrivialHarmonics();
};

#include "operator.ipp"
#endif