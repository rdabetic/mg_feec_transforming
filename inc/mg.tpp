#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mfem.hpp"
#include "operator.tpp"
#include "qrsolve.hpp"
#include "transfer.hpp"

// =======================================================================
//                    DEC OPERATOR WRAPPER DECLARATION
// =======================================================================

/**
 * @class DecWrapper
 * @brief Wraps a specific operator to easily apply it within the MFEM solver
 * framework.
 */
template <typename OpType, unsigned int... FORMS>
class DecWrapper : public mfem::Operator
{
private:
    std::shared_ptr<const OpType> op_;
    mfem::Array<int>              offsets_;

public:
    explicit DecWrapper(std::shared_ptr<const OpType> op);

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    [[nodiscard]] const mfem::Array<int>& getOffsets() const
    {
        return offsets_;
    }
};

// =======================================================================
//                    MULTIGRID CONFIGURATION
// =======================================================================

/**
 * @brief Defines the Multigrid cycle pattern.
 */
enum class CycleType
{
    VCYCLE,  ///< Standard V-cycle.
    WCYCLE   ///< Standard W-cycle.
};

// =======================================================================
//                    CUSTOM MULTIGRID SOLVER DECLARATION
// =======================================================================

/**
 * @class Multigrid
 * @brief A generic geometric multigrid solver framework for Whitney forms.
 *
 * @tparam DerivedOperator The concrete operator type being solved.
 * @tparam SmootherType The type of smoother to use (e.g., Jacobi,
 * Gauss-Seidel).
 * @tparam ACTIVE_FORMS The degrees of forms involved in the operator.
 */
template <
    class DerivedOperator,
    typename SmootherType,
    unsigned int... ACTIVE_FORMS>
class Multigrid : public mfem::Solver
{
public:
    using OperatorType = WhitneyFormOperator<DerivedOperator, ACTIVE_FORMS...>;
    using WrapperType  = DecWrapper<OperatorType, ACTIVE_FORMS...>;

protected:
    OperatorMode mg_mode_;
    CycleType    cycle_type_;
    int          pre_smooth_iters_            = 1;
    int          post_smooth_iters_           = 1;
    bool         eliminate_trivial_harmonics_ = true;

    // Hierarchy levels (index 0 is coarsest, back() is finest)
    std::vector<std::shared_ptr<DerivedOperator>>     operators_;
    std::vector<std::unique_ptr<WrapperType>>         dec_wrappers_;
    std::vector<std::unique_ptr<mfem::BlockOperator>> prolongations_;
    std::vector<std::unique_ptr<SmootherType>>        smoothers_;

    std::shared_ptr<mfem::Solver> coarse_solver_;

public:
    /**
     * @brief Multigrid constructor.
     * @param mode Operator mode (Galerkin/DEC).
     */
    explicit Multigrid(const OperatorMode mode = OperatorMode::Galerkin);

    // =====================================================================
    //                   CONFIGURATION GETTERS & SETTERS
    // =====================================================================

    /**
     * @brief Set solver to iterative mode (accumulates into x rather than
     * overwriting).
     */
    void setIterativeMode(const bool mode) { iterative_mode = mode; }
    [[nodiscard]] bool getIterativeMode() const { return iterative_mode; }

    void setEliminateTrivialHarmonics(const bool elim)
    {
        eliminate_trivial_harmonics_ = elim;
    }
    [[nodiscard]] bool getEliminateTrivialHarmonics() const
    {
        return eliminate_trivial_harmonics_;
    }

    void setMode(const OperatorMode mode)
    {
        mg_mode_ = mode;
        if (!operators_.empty())
            for (auto& op : operators_)
                if (op)
                    op->setMode(mode);
    }
    [[nodiscard]] OperatorMode getMode() const { return mg_mode_; }

    void setCycleType(const CycleType type) { cycle_type_ = type; }
    [[nodiscard]] CycleType getCycleType() const { return cycle_type_; }

    void setSmoothingIterations(const int pre_iters, const int post_iters)
    {
        pre_smooth_iters_  = pre_iters;
        post_smooth_iters_ = post_iters;
    }
    [[nodiscard]] int getPreSmoothingIterations() const
    {
        return pre_smooth_iters_;
    }
    [[nodiscard]] int getPostSmoothingIterations() const
    {
        return post_smooth_iters_;
    }

    [[nodiscard]] std::size_t numLevels() const { return operators_.size(); }

    void SetOperator(const mfem::Operator& op) override {}

    // =====================================================================
    //                   HIERARCHY BUILDING
    // =====================================================================

    /**
     * @brief Appends a new level to the multigrid hierarchy.
     * @param fine_op Shared pointer to the new finest operator.
     * @param args Arguments forwarded to construct the smoother for this level.
     */
    template <typename... SMOOTHER_ARGS>
    void addLevel(
        std::shared_ptr<DerivedOperator> fine_op,
        SMOOTHER_ARGS... args);

    /**
     * @brief Automatically generates and appends a refined level using the
     * current finest operator.
     * @param args Arguments forwarded to construct the smoother.
     */
    template <typename... SMOOTHER_ARGS>
    void addRefinedLevel(SMOOTHER_ARGS... args);

    /**
     * @brief Sets up the coarsest level solver explicitly.
     * @param coarse_solver Pre-configured solver to use for the coarse grid.
     * If nullptr, defaults to QR
     */
    template <typename... OPERATOR_ARGS>
    void addCoarseLevel(
        std::shared_ptr<mfem::Solver> coarse_solver,
        OPERATOR_ARGS... args);

    // =====================================================================
    //                   OPERATOR ACCESSORS
    // =====================================================================

    [[nodiscard]] std::shared_ptr<const DerivedOperator> getOperator(
        const size_t level) const
    {
        return operators_.at(level);
    }

    [[nodiscard]] std::shared_ptr<const DerivedOperator> getFinestOperator()
        const
    {
        if (operators_.empty())
            throw std::logic_error("Hierarchy is empty.");
        return operators_.back();
    }

    [[nodiscard]] std::shared_ptr<DerivedOperator> getFinestOperator()
    {
        if (operators_.empty())
            throw std::logic_error("Hierarchy is empty.");
        return operators_.back();
    }

    // =====================================================================
    //                   SOLVER EXECUTION
    // =====================================================================

    /**
     * @brief Executes a single multigrid cycle.
     * @param b Right hand side vector.
     * @param x Approximate solution vector (updated in place).
     */
    void Mult(const mfem::Vector& b, mfem::Vector& x) const override;
    void MultTranspose(const mfem::Vector& b, mfem::Vector& x) const override;

protected:
    void mgCycle(const int level, const mfem::Vector& b, mfem::Vector& x) const;
    void createCoarseSolver();

    [[nodiscard]] std::unique_ptr<mfem::BlockOperator> buildBlockProlongation(
        const OperatorType& coarse_op,
        const OperatorType& fine_op) const;
};

#include "mg.ipp"
#endif
