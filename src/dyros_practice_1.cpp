#include <pinocchio/fwd.hpp>
#include "ros/ros.h"
#include <fstream>
#include <string>
#include <vector>
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/model.hpp"
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/frame.hpp>
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/centroidal-derivatives.hxx"
#include "pinocchio/algorithm/aba-derivatives.hxx"
#include "pinocchio/parsers/sample-models.hpp"

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/box-ddp.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "pinocchio/parsers/urdf.hpp"
//#include <pinocchio/autodiff/casadi.hpp>
#include <string.h>
#define MODEL_DOF 33
#define ENDEFFECTOR_NUMBER 4
#define LINK_NUMBER 34
#define MODEL_DOF_VIRTUAL 39
#define MODEL_DOF_QVIRTUAL 40

#define GRAVITY 9.80665
#define MAX_DOF 50U
#define RAD2DEG 1 / DEG2RAD

#define INERITA_SIZE 198
#define STDDEV(vec) std::sqrt(((vec - vec.mean())).square().sum() / (static_cast<double>(vec.size()) - 1))
#define AVG(vec) (vec.mean())
namespace Eigen
{
  typedef double rScalar;
  typedef float lScalar;
  typedef Matrix<rScalar, MODEL_DOF, 1> VectorQd;
  typedef Matrix<rScalar, MODEL_DOF_VIRTUAL, 1> VectorVQd;
  typedef Matrix<rScalar, MODEL_DOF_QVIRTUAL, 1> VectorQVQd;
}
using namespace std;

pinocchio::Model model;
pinocchio::Data model_data;

pinocchio::Model model1;
pinocchio::Data model_data1;

namespace crocoddyl
{
  template <typename _Scalar>
  struct DifferentialActionDataKinoDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef DifferentialActionDataAbstractTpl<Scalar> Base;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar> class Model>
    explicit DifferentialActionDataKinoDynamicsTpl(Model<Scalar> *const model)
        : Base(model),
          pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
          multibody(&pinocchio, model->get_actuation()->createData()),
          costs(model->get_costs()->createData(&multibody)),
          Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
          u_drift(model->get_nu()),
          dtau_dx(model->get_nu(), model->get_state()->get_ndx()),
          tmp_xstatic(model->get_state()->get_nx())
    {
      costs->shareMemory(this);
      Minv.setZero();
      u_drift.setZero();
      dtau_dx.setZero();
      tmp_xstatic.setZero();
    }

    pinocchio::DataTpl<Scalar> pinocchio;
    DataCollectorActMultibodyTpl<Scalar> multibody;
    boost::shared_ptr<CostDataSumTpl<Scalar>> costs;
    MatrixXs Minv;
    VectorXs u_drift;
    MatrixXs dtau_dx;
    VectorXs tmp_xstatic;

    using Base::cost;
    using Base::Fu;
    using Base::Fx;
    using Base::Lu;
    using Base::Luu;
    using Base::Lx;
    using Base::Lxu;
    using Base::Lxx;
    using Base::r;
    using Base::xout;
    using Base::xout2;
    // using Base::dhg;
  };

} // namespace crocoddyl

namespace crocoddyl
{

  /**
   * @brief State Kinodynamic representation
   *
   * A Kinodynamic state is described by the configuration point and its tangential velocity, or in other words, by the
   * generalized position and velocity coordinates of a rigid-body system. For this state, we describe its operators:
   * difference, integrates, transport and their derivatives for any Pinocchio model.
   *
   * For more details about these operators, please read the documentation of the `StateAbstractTpl` class.
   *
   * \sa `diff()`, `integrate()`, `Jdiff()`, `Jintegrate()` and `JintegrateTransport()`
   */
  template <typename _Scalar>
  class StateKinodynamicTpl : public StateAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef StateAbstractTpl<Scalar> Base;
    typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the Kinodynamic state
     *
     * @param[in] model  Pinocchio model
     */
    explicit StateKinodynamicTpl(boost::shared_ptr<PinocchioModel> model);
    StateKinodynamicTpl();
    virtual ~StateKinodynamicTpl();

    /**
     * @brief Generate a zero state.
     *
     * Note that the zero configuration is computed using `pinocchio::neutral`.
     */
    virtual VectorXs zero() const;

    /**
     * @brief Generate a random state
     *
     * Note that the random configuration is computed using `pinocchio::random` which satisfies the manifold definition
     * (e.g., the quaterion definition)
     */
    virtual VectorXs rand() const;

    virtual void diff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                      Eigen::Ref<VectorXs> dxout) const;
    virtual void diff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                       Eigen::Ref<VectorXs> dxout) const;
    virtual void integrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                           Eigen::Ref<VectorXs> xout) const;
    virtual void Jdiff(const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &, Eigen::Ref<MatrixXs> Jfirst,
                       Eigen::Ref<MatrixXs> Jsecond, const Jcomponent firstsecond = both) const;
    virtual void Jdiff1(const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &, Eigen::Ref<MatrixXs> Jfirst,
                        Eigen::Ref<MatrixXs> Jsecond, const Jcomponent firstsecond = both) const;
    virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                            Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                            const Jcomponent firstsecond = both, const AssignmentOp = setto) const;
    virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                     Eigen::Ref<MatrixXs> Jin, const Jcomponent firstsecond) const;

    /**
     * @brief Return the Pinocchio model (i.e., model of the rigid body system)
     */
    const boost::shared_ptr<PinocchioModel> &get_pinocchio() const;

  protected:
    using Base::has_limits_;
    using Base::lb_;
    using Base::ndx_;
    using Base::nq_;
    using Base::nv_;
    using Base::nx_;
    using Base::ub_;

  private:
    boost::shared_ptr<PinocchioModel> pinocchio_; //!< Pinocchio model
    VectorXs x0_;                                 //!< Zero state
  };

} // namespace crocoddyl

namespace crocoddyl
{

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::StateKinodynamicTpl(boost::shared_ptr<PinocchioModel> model)
      : Base(model->nq + model->nv, 2 * model->nv), pinocchio_(model), x0_(VectorXs::Zero(model->nq + model->nv + 8))
  {

    const std::size_t nq0 = model->joints[1].nq();
    x0_.head(nq_) = pinocchio::neutral(*pinocchio_.get());

    // In a Kinodynamic system, we could define the first joint using Lie groups.
    // The current cases are free-flyer (SE3) and spherical (S03).
    // Instead simple represents any joint that can model within the Euclidean manifold.
    // The rest of joints use Euclidean algebra. We use this fact for computing Jdiff.

    // Define internally the limits of the first joint

    lb_.head(nq0) = -3.14 * VectorXs::Ones(nq0);
    ub_.head(nq0) = 3.14 * VectorXs::Ones(nq0);
    lb_.segment(nq0, nq_ - nq0) = pinocchio_->lowerPositionLimit.tail(nq_ - nq0);
    ub_.segment(nq0, nq_ - nq0) = pinocchio_->upperPositionLimit.tail(nq_ - nq0);
    lb_.segment(nq_, nv_) = -pinocchio_->velocityLimit;
    ub_.segment(nq_, nv_) = pinocchio_->velocityLimit;
    lb_.tail(8).head(3) = -1.0 * VectorXs::Ones(3);
    ub_.tail(8).head(3) = 1.0 * VectorXs::Ones(3);
    lb_.tail(4).head(3) = -1.0 * VectorXs::Ones(3);
    ub_.tail(4).head(3) = 1.0 * VectorXs::Ones(3);
    lb_.tail(5).head(1) = -5.0 * VectorXs::Ones(1);
    ub_.tail(5).head(1) = 5.0 * VectorXs::Ones(1);
    lb_.tail(1) = -5.0 * VectorXs::Ones(1);
    ub_.tail(1) = 5.0 * VectorXs::Ones(1);
    Base::update_has_limits();
  }

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::StateKinodynamicTpl() : Base(), x0_(VectorXs::Zero(0)) {}

  template <typename Scalar>
  StateKinodynamicTpl<Scalar>::~StateKinodynamicTpl() {}

  template <typename Scalar>
  typename MathBaseTpl<Scalar>::VectorXs StateKinodynamicTpl<Scalar>::zero() const
  {
    return x0_;
  }

  template <typename Scalar>
  typename MathBaseTpl<Scalar>::VectorXs StateKinodynamicTpl<Scalar>::rand() const
  {
    VectorXs xrand = VectorXs::Random(nx_);
    xrand.head(nq_) = pinocchio::randomConfiguration(*pinocchio_.get());
    return xrand;
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::diff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                         Eigen::Ref<VectorXs> dxout) const
  {
    // std::cout << "diff " << x0.tail(4).transpose() << std::endl;
    // std::cout << "diffx " << x1.tail(4).transpose() << std::endl;

    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dxout.size()) != ndx_)
    {
      throw_pretty("Invalid argument: "
                   << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }
    pinocchio::difference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), dxout.head(nv_));
    dxout.segment(nq_, nv_) = x1.segment(nq_, nv_) - x0.segment(nq_, nv_);
    dxout.tail(8) = x1.tail(8) - x0.tail(8);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::diff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                          Eigen::Ref<VectorXs> dxout) const
  {
    // std::cout << "diff " << x0.tail(4).transpose() << std::endl;
    // std::cout << "diffx " << x1.tail(4).transpose() << std::endl;
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(dxout.size()) != 2)
    {
      throw_pretty("Invalid argument: "
                   << "dxout has wrong dimension (it should be " + std::to_string(ndx_) + ")");
    }

    dxout.setZero();
    dxout.head(1) = x1.tail(6).head(1) - x0.tail(6).head(1);
    dxout.tail(1) = x1.tail(2).head(1) - x0.tail(2).head(1);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::integrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                              Eigen::Ref<VectorXs> xout) const
  {
    if (static_cast<std::size_t>(x.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    pinocchio::integrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), xout.head(nq_));
    xout.segment(nq_, nv_) = x.segment(nq_, nv_) + dx.segment(nq_ - 1, nv_);
    xout.tail(8) = x.tail(8) + dx.tail(8);
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jdiff(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                          Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                          const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    if (firstsecond == first)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }

      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
    }
    else if (firstsecond == second)
    {
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
    else
    { // computing both
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
  }
  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jdiff1(const Eigen::Ref<const VectorXs> &x0, const Eigen::Ref<const VectorXs> &x1,
                                           Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                           const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    if (static_cast<std::size_t>(x0.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x0 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }
    if (static_cast<std::size_t>(x1.size()) != nx_ + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x1 has wrong dimension (it should be " + std::to_string(nx_) + ")");
    }

    if (firstsecond == first)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }

      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
    }
    else if (firstsecond == second)
    {

      // pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
      // pinocchio::ARG1);
      // Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      // Jsecond.setIdentity();
      Jsecond.setZero();
      Jsecond.bottomRightCorner(2, 6).topLeftCorner(1, 1).diagonal().array() = (Scalar)1;
      Jsecond.bottomRightCorner(2, 2).bottomLeftCorner(1, 1).diagonal().array() = (Scalar)1;
      // Jsecond.bottomRightCorner(2,2).topLefFtCorner(1,1).diagonal().array() = (Scalar)1;
      // Jsecond.bottomRightCorner(6,6).topLeftCorner(1,1).diagonal().array() = (Scalar)1;
    }
    else
    { // computing both
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jfirst.topLeftCorner(nv_, nv_),
                             pinocchio::ARG0);
      pinocchio::dDifference(*pinocchio_.get(), x0.head(nq_), x1.head(nq_), Jsecond.topLeftCorner(nv_, nv_),
                             pinocchio::ARG1);
      Jfirst.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)-1;
      Jsecond.block(nv_, nv_, nv_, nv_).diagonal().array() = (Scalar)1;
      Jfirst.bottomRightCorner(8, 8).diagonal().array() = (Scalar)-1;
      Jsecond.bottomRightCorner(8, 8).diagonal().array() = (Scalar)1;
    }
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::Jintegrate(const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &dx,
                                               Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                                               const Jcomponent firstsecond, const AssignmentOp op) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));
    assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));

    if (firstsecond == first || firstsecond == both)
    {
      if (static_cast<std::size_t>(Jfirst.rows()) != ndx_ || static_cast<std::size_t>(Jfirst.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jfirst has wrong dimension (it should be " + std::to_string(ndx_) + "," + std::to_string(ndx_) +
                            ")");
      }
      switch (op)
      {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::SETTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::ADDTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jfirst.topLeftCorner(nv_, nv_),
                              pinocchio::ARG0, pinocchio::RMTO);
        Jfirst.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
    if (firstsecond == second || firstsecond == both)
    {
      if (static_cast<std::size_t>(Jsecond.rows()) != ndx_ || static_cast<std::size_t>(Jsecond.cols()) != ndx_)
      {
        throw_pretty("Invalid argument: "
                     << "Jsecond has wrong dimension (it should be " + std::to_string(ndx_) + "," +
                            std::to_string(ndx_) + ")");
      }
      switch (op)
      {
      case setto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::SETTO);
        Jsecond.setZero();
        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() = (Scalar)1;
        break;
      case addto:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::ADDTO);
        Jsecond.setZero();
        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() += (Scalar)1;
        break;
      case rmfrom:
        pinocchio::dIntegrate(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jsecond.topLeftCorner(nv_, nv_),
                              pinocchio::ARG1, pinocchio::RMTO);
        Jsecond.setZero();

        Jsecond.bottomRightCorner(nv_ + 8, nv_ + 8).diagonal().array() -= (Scalar)1;
        break;
      default:
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  template <typename Scalar>
  void StateKinodynamicTpl<Scalar>::JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                                        const Eigen::Ref<const VectorXs> &dx, Eigen::Ref<MatrixXs> Jin,
                                                        const Jcomponent firstsecond) const
  {
    assert_pretty(is_a_Jcomponent(firstsecond), ("firstsecond must be one of the Jcomponent {both, first, second}"));

    switch (firstsecond)
    {
    case first:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG0);
      break;
    case second:
      pinocchio::dIntegrateTransport(*pinocchio_.get(), x.head(nq_), dx.head(nv_), Jin.topRows(nv_), pinocchio::ARG1);
      break;
    default:
      throw_pretty(
          "Invalid argument: firstsecond must be either first or second. both not supported for this operation.");
      break;
    }
  }

  template <typename Scalar>
  const boost::shared_ptr<pinocchio::ModelTpl<Scalar>> &StateKinodynamicTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

} // namespace crocoddyl

namespace crocoddyl
{

  /**
   * @brief Floating-base actuation model
   *
   * It considers the first joint, defined in the Pinocchio model, as the floating-base joints.
   * Then, this joint (that might have various DoFs) is unactuated.
   *
   * The main computations are carrying out in `calc`, and `calcDiff`, where the former computes actuation signal
   * \f$\mathbf{a}\f$ from a given control input \f$\mathbf{u}\f$ and state point \f$\mathbf{x}\f$, and the latter
   * computes the Jacobians of the actuation-mapping function. Note that `calcDiff` requires to run `calc` first.
   *
   * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  template <typename _Scalar>
  class ActuationModelFloatingKinoBaseTpl : public ActuationModelAbstractTpl<_Scalar>
  {
  public:
    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ActuationModelAbstractTpl<Scalar> Base;
    typedef ActuationDataAbstractTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the floating-base actuation model
     *
     * @param[in] state  State of a multibody system
     * @param[in] nu     Dimension of control vector
     */
    explicit ActuationModelFloatingKinoBaseTpl(boost::shared_ptr<StateKinodynamic> state)
        : Base(state, state->get_nv()){};
    virtual ~ActuationModelFloatingKinoBaseTpl(){};

    /**
     * @brief Compute the floating-base actuation signal from the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     *
     * @param[in] data  Actuation data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<Data> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u)
    {
      if (static_cast<std::size_t>(u.size()) != nu_ + 4)
      {
        throw_pretty("Invalid argument: "
                     << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
      }
      data->tau.segment(0, nu_) = u.head(nu_);
      data->u_x = u.tail(4);
    };

      /**
       * @brief Compute the Jacobians of the floating-base actuation function
       *
       * @param[in] data  Actuation data
       * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
       * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
       */
#ifndef NDEBUG
    virtual void calcDiff(const boost::shared_ptr<Data> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u)
    {
#else
    virtual void calcDiff(const boost::shared_ptr<Data> &, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u)
    {
#endif
      // The derivatives has constant values which were set in createData.
      assert_pretty(data->dtau_dx.isZero(), "dtau_dx has wrong value");
      assert_pretty(MatrixXs(data->dtau_du).isApprox(dtau_du_), "dtau_du has wrong value");
    };

    /**
     * @brief Create the floating-base actuation data
     *
     * @return the actuation data
     */
    virtual boost::shared_ptr<Data> createData()
    {
      typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
      boost::shared_ptr<StateKinodynamic> state = boost::static_pointer_cast<StateKinodynamic>(state_);
      boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
      data->dtau_du.diagonal(nu_).setOnes();
#ifndef NDEBUG
      dtau_du_ = data->dtau_du;
#endif
      return data;
    };

  protected:
    using Base::nu_;
    using Base::state_;

#ifndef NDEBUG
  private:
    MatrixXs dtau_du_;
#endif
  };

} // namespace crocoddyl

namespace crocoddyl
{
  template <typename _Scalar>
  class DifferentialActionModelKinoDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef DifferentialActionModelAbstractTpl<Scalar> Base;
    typedef DifferentialActionDataKinoDynamicsTpl<Scalar> Data;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef CostModelSumTpl<Scalar> CostModelSum;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
    typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    DifferentialActionModelKinoDynamicsTpl(boost::shared_ptr<StateKinodynamic> state,
                                           boost::shared_ptr<ActuationModelAbstract> actuation,
                                           boost::shared_ptr<CostModelSum> costs);
    virtual ~DifferentialActionModelKinoDynamicsTpl();

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);

    virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x);

    virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

    virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract> &data);

    virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract> &data, Eigen::Ref<VectorXs> u,
                             const Eigen::Ref<const VectorXs> &x, const std::size_t maxiter = 100,
                             const Scalar tol = Scalar(1e-9));

    const boost::shared_ptr<ActuationModelAbstract> &get_actuation() const;

    const boost::shared_ptr<CostModelSum> &get_costs() const;

    pinocchio::ModelTpl<Scalar> &get_pinocchio() const;

    const VectorXs &get_armature() const;

    void set_armature(const VectorXs &armature);

    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;    //!< Control dimension
    using Base::state_; //!< Model of the state

  private:
    boost::shared_ptr<ActuationModelAbstract> actuation_; //!< Actuation model
    boost::shared_ptr<CostModelSum> costs_;               //!< Cost model
    pinocchio::ModelTpl<Scalar> &pinocchio_;              //!< Pinocchio model
    bool without_armature_;                               //!< Indicate if we have defined an armature
    VectorXs armature_;                                   //!< Armature vector
  };
}

namespace crocoddyl
{

  template <typename Scalar>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::DifferentialActionModelKinoDynamicsTpl(
      boost::shared_ptr<StateKinodynamic> state, boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs)
      : Base(state, actuation->get_nu(), costs->get_nr()),
        actuation_(actuation),
        costs_(costs),
        pinocchio_(*state->get_pinocchio().get()),
        without_armature_(true),
        armature_(VectorXs::Zero(state->get_nv()))
  {
    if (costs_->get_nu() != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
    }
    VectorXs temp;
    temp.resize(actuation->get_nu() + 4);
    temp.setZero();
    temp.head(nu_) = pinocchio_.effortLimit.head(nu_);
    temp(nu_) = 10;
    temp(nu_ + 1) = 10;
    temp(nu_ + 2) = 10;
    temp(nu_ + 3) = 10;
    Base::set_u_lb(Scalar(-1.) * temp);
    Base::set_u_ub(Scalar(+1.) * temp);
  }

  template <typename Scalar>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::~DifferentialActionModelKinoDynamicsTpl() {}

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
      const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }

    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());

    actuation_->calc(d->multibody.actuation, x, u);

    // Computing the dynamics using ABA or manually for armature case
    /* if (without_armature_) {
       d->xout = pinocchio::aba(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau.segment(0,state_->get_nv()));
       pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
     } else {

       pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
       d->pinocchio.M.diagonal() += armature_;
       pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
       d->Minv.setZero();
       pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);
       d->u_drift = d->multibody.actuation->tau - d->pinocchio.nle;
       d->xout.noalias() = d->Minv * d->u_drift;
     }*/

    d->xout = d->multibody.actuation->tau;
    d->xout = d->multibody.actuation->tau.segment(0, state_->get_nv());
    d->xout2 << x_state[1], 6.59308329 * x_state[0] - 6.59308329 * x_state[2] - d->multibody.actuation->u_x[1] * 1.0 / 50.0, d->multibody.actuation->u_x[0], d->multibody.actuation->u_x[1], x_state[5], 6.59308329 * x_state[4] - 6.59308329 * x_state[6] + d->multibody.actuation->u_x[3] * 1.0 / 50.0, d->multibody.actuation->u_x[2], d->multibody.actuation->u_x[3]; // d->dhg;

    // Computing the cost value and residuals
    costs_->calc(d->costs, x, u);
    d->cost = d->costs->cost;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    Data *d = static_cast<Data *>(data.get());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),state_->get_nv());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(4);

    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
      const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }

    const std::size_t nv = state_->get_nv();
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),nv);
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());
    // onst Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(4);

    Data *d = static_cast<Data *>(data.get());
    actuation_->calcDiff(d->multibody.actuation, x, u);

    d->Fx.bottomRightCorner(8, 8).topLeftCorner(4, 4) << 0.0, 1.0, 0.0, 0.0, 6.59308329, 0.0, -6.59308329, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    d->Fx.bottomRightCorner(4, 4) << 0.0, 1.0, 0.0, 0.0, 6.59308329, 0.0, -6.59308329, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // d->Fx.block(0, state_->get_nv(), state_->get_nv(), state_->get_nv()).setIdentity();
    d->Fu.topLeftCorner(nu_, nu_).setIdentity();
    d->Fu.bottomRightCorner(8, 4).topLeftCorner(4, 2) << 0.0, 0.0, 0.0, -1.0 / 50.0, 1.0, 0.0, 0.0, 1.0;
    d->Fu.bottomRightCorner(4, 2) << 0.0, 0.0, 0.0, 1.0 / 50.0, 1.0, 0.0, 0.0, 1.0;

    /*
      std::cout << "d->Fx" << std::endl;
      std::cout << d->Fx << std::endl;

      std::cout << "d->Fu" << std::endl;
      std::cout << d->Fu << std::endl;
    */
    costs_->calcDiff(d->costs, x, u);
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    Data *d = static_cast<Data *>(data.get());
    costs_->calcDiff(d->costs, x);
  }

  template <typename Scalar>
  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::createData()
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  template <typename Scalar>
  bool DifferentialActionModelKinoDynamicsTpl<Scalar>::checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data)
  {
    boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
    if (d != NULL)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::quasiStatic(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, Eigen::Ref<VectorXs> u,
      const Eigen::Ref<const VectorXs> &x, const std::size_t, const Scalar)
  {
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // Static casting the data
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

    const std::size_t nq = state_->get_nq();
    const std::size_t nv = state_->get_nv();

    // Check the velocity input is zero
    assert_pretty(x.segment(nq, nv).isZero(), "The velocity input should be zero for quasi-static to work.");

    d->tmp_xstatic.head(nq) = q;
    d->tmp_xstatic.segment(nq, nv).setZero();
    u.setZero();

    pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.segment(nq, nv), d->tmp_xstatic.segment(nq, nv));
    actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
    actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);

    u.noalias() = pseudoInverse(d->multibody.actuation->dtau_du) * d->pinocchio.tau;
    d->pinocchio.tau.setZero();
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::print(std::ostream &os) const
  {
    os << "DifferentialActionModelKinoDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
       << ", nu=" << nu_ << "}";
  }

  template <typename Scalar>
  pinocchio::ModelTpl<Scalar> &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ActuationModelAbstractTpl<Scalar>> &
  DifferentialActionModelKinoDynamicsTpl<Scalar>::get_actuation() const
  {
    return actuation_;
  }

  template <typename Scalar>
  const boost::shared_ptr<CostModelSumTpl<Scalar>> &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_costs()
      const
  {
    return costs_;
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_armature() const
  {
    return armature_;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::set_armature(const VectorXs &armature)
  {
    if (static_cast<std::size_t>(armature.size()) != state_->get_nv())
    {
      throw_pretty("Invalid argument: "
                   << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
    }

    armature_ = armature;
    without_armature_ = false;
  }

} // namespace crocoddyl

namespace crocoddyl
{
  template <typename _Scalar>
  struct ResidualDataCentroidalAngularMomentumTpl : public ResidualDataAbstractTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Matrix6xs Matrix6xs;

    template <template <typename Scalar> class Model>
    ResidualDataCentroidalAngularMomentumTpl(Model<Scalar> *const model, DataCollectorAbstract *const data)
        : Base(model, data), dhd_dq(6, model->get_state()->get_nv()), dhd_dv(6, model->get_state()->get_nv()), dhd_da(6, model->get_state()->get_nv()), dh_dq(6, model->get_state()->get_nv())
    {
      dhd_dq.setZero();
      dhd_dv.setZero();

      // Check that proper shared data has been passed
      DataCollectorMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorMultibodyTpl<Scalar> *>(shared);
      if (d == NULL)
      {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
      }

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
    }

    pinocchio::DataTpl<Scalar> *pinocchio; //!< Pinocchio data
    Matrix6xs dhd_dq;                      //!< Jacobian of the centroidal momentum
    Matrix6xs dhd_dv;                      //!< Jacobian of the centroidal momentum
    Matrix6xs dh_dq;                       //!< Jacobian of the centroidal momentum
    Matrix6xs dhd_da;                      //!< Jacobian of the centroidal momentum

    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

} // namespace crocoddyl

namespace crocoddyl
{
  template <typename _Scalar>
  class ResidualModelCentroidalAngularMomentumTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataCentroidalAngularMomentumTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Vector6s Vector6s;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::Matrix6xs Matrix6xs;

    /**
     * @brief Initialize the centroidal momentum residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] href   Reference centroidal momentum
     * @param[in] nu     Dimension of the control vector
     */
    ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state, const Vector6s &href,
                                              const std::size_t nu);

    /**
     * @brief Initialize the centroidal momentum residual model
     *
     * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state  State of the multibody system
     * @param[in] href   Reference centroidal momentum
     */
    ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state, const std::size_t nu);
    virtual ~ResidualModelCentroidalAngularMomentumTpl();

    /**
     * @brief Compute the centroidal momentum residual
     *
     * @param[in] data  Centroidal momentum residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the centroidal momentum residual
     *
     * @param[in] data  Centroidal momentum residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Create the centroidal momentum residual data
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

    /**
     * @brief Return the reference centroidal momentum
     */
    const Vector6s &get_reference() const;

    /**
     * @brief Modify the reference centroidal momentum
     */
    void set_reference(const Vector6s &href);

    /**
     * @brief Print relevant information of the centroidal-momentum residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;

  private:
    Vector6s href_;                                                          //!< Reference centroidal momentum
    boost::shared_ptr<typename StateKinodynamic::PinocchioModel> pin_model_; //!< Pinocchio model
  };
}

namespace crocoddyl
{

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                                               const Vector6s &href,
                                                                                               const std::size_t nu)
      : Base(state, 2, nu, true, true, true, false, false), href_(href), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                                               const std::size_t nu)
      : Base(state, 2, nu, true, true, true, false, false), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::~ResidualModelCentroidalAngularMomentumTpl() {}

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                               const Eigen::Ref<const VectorXs> &x,
                                                               const Eigen::Ref<const VectorXs> &u)
  {
    // Compute the residual residual give the reference centroidal momentum
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> u_state = u.tail(4);
    pinocchio::computeCentroidalMomentum(*pin_model_.get(), *d->pinocchio, q, v);
    data->r(0) = d->pinocchio->hg.toVector()(3) - x_state(7);
    data->r(1) = d->pinocchio->hg.toVector()(4) - x_state(3);
    // std::cout << "data_>r" << std::endl;
    // std::cout << data->r(0) <<" " << data->r(1)<< std::endl;
  }

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                                   const Eigen::Ref<const VectorXs> &x,
                                                                   const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());
    const std::size_t &nv = state_->get_nv();
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());
    pinocchio::computeRNEADerivatives(*pin_model_.get(), *d->pinocchio, q, v, a);
    pinocchio::getCentroidalDynamicsDerivatives(*pin_model_.get(), *d->pinocchio, d->dh_dq, d->dhd_dq, d->dhd_dv, d->dhd_da);
    data->Rx.rightCols(1)(0) = -1;
    data->Rx.rightCols(5).leftCols(1)(1) = -1;
    data->Rx.leftCols(nv) = d->dh_dq.block(3, 0, 2, nv);
    data->Rx.block(0, nv, 2, nv) = d->dhd_da.block(3, 0, 2, nv);
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualModelCentroidalAngularMomentumTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualModelCentroidalAngularMomentum {href=" << href_.transpose().format(fmt) << "}";
  }
} // namespace crocoddyl

namespace crocoddyl
{
  template <typename _Scalar>
  class ResidualModelCoMKinoPositionTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataCoMPositionTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Vector3s Vector3s;
    typedef typename MathBase::VectorXs VectorXs;

    /**
     * @brief Initialize the CoM position residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] cref   Reference CoM position
     * @param[in] nu     Dimension of the control vector
     */
    ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state, const std::size_t nu);

    /**
     * @brief Initialize the CoM position residual model
     *
     * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state  State of the multibody system
     * @param[in] cref   Reference CoM position
     */
    ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state);
    virtual ~ResidualModelCoMKinoPositionTpl();

    /**
     * @brief Compute the CoM position residual
     *
     * @param[in] data  CoM position residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the CoM position residual
     *
     * @param[in] data  CoM position residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

    /**
     * @brief Print relevant information of the com-position residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;
    using Base::v_dependent_;

  private:
    Vector3s cref_;                                                          //!< Reference CoM position
    boost::shared_ptr<typename StateKinodynamic::PinocchioModel> pin_model_; //!< Pinocchio model
  };

  template <typename _Scalar>
  struct ResidualDataCoMPositionTpl : public ResidualDataAbstractTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Matrix3xs Matrix3xs;

    template <template <typename Scalar> class Model>
    ResidualDataCoMPositionTpl(Model<Scalar> *const model, DataCollectorAbstract *const data) : Base(model, data)
    {
      // Check that proper shared data has been passed
      DataCollectorMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorMultibodyTpl<Scalar> *>(shared);
      if (d == NULL)
      {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
      }

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
    }

    pinocchio::DataTpl<Scalar> *pinocchio; //!< Pinocchio data
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

} // namespace crocoddyl

namespace crocoddyl
{

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state, const std::size_t nu)
      : Base(state, 3, nu, true, false, true, false, false), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state)
      : Base(state, 3, true, false, true, false, false), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::~ResidualModelCoMKinoPositionTpl() {}

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                     const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    // Compute the residual residual give the reference CoMPosition position
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    pinocchio::centerOfMass(*pin_model_.get(), *d->pinocchio, q, false);
    data->r(0) = d->pinocchio->com[0](0) - x_state(0);
    data->r(1) = d->pinocchio->com[0](1) - x_state(4);
    data->r(2) = d->pinocchio->com[0](2) - 5.11307390e-01;
  }

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                         const Eigen::Ref<const VectorXs> &x,
                                                         const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

    pinocchio::jacobianCenterOfMass(*pin_model_.get(), *d->pinocchio, q, false);

    // Compute the derivatives of the frame placement
    const std::size_t nv = state_->get_nv();
    data->Rx.leftCols(nv) = d->pinocchio->Jcom.block(0, 0, 3, nv);
    (data->Rx.rightCols(8)).leftCols(1)(0) = -1.0;
    (data->Rx.rightCols(4)).leftCols(1)(1) = -1.0;
    //(data->Rx.rightCols(4)).leftCols(1) = -1 * (data->Rx.rightCols(4)).leftCols(1);
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualModelCoMKinoPositionTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualModelCoMPosition {cref=" << cref_.transpose().format(fmt) << "}";
  }
} // namespace crocoddyl

namespace crocoddyl
{

  /**
   * @brief State residual
   *
   * This residual function defines the state tracking as \f$\mathbf{r}=\mathbf{x}\ominus\mathbf{x}^*\f$, where
   * \f$\mathbf{x},\mathbf{x}^*\in~\mathcal{X}\f$ are the current and reference states, respectively, which belong to the
   * state manifold \f$\mathcal{X}\f$. Note that the dimension of the residual vector is obtained from
   * `StateAbstract::get_ndx()`. Furthermore, the Jacobians of the residual function are
   * computed analytically.
   *
   * As described in `ResidualModelAbstractTpl()`, the residual value and its derivatives are calculated by `calc` and
   * `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  template <typename _Scalar>
  class ResidualFlyStateTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the state residual model
     *
     * @param[in] state       State of the multibody system
     * @param[in] xref        Reference state
     * @param[in] nu          Dimension of the control vector
     */
    ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs &xref,
                        const std::size_t nu);

    /**
     * @brief Initialize the state residual model
     *
     * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state       State of the multibody system
     * @param[in] xref        Reference state
     */
    ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs &xref);

    /**
     * @brief Initialize the state residual model
     *
     * The default reference state is obtained from `StateAbstractTpl::zero()`.
     *
     * @param[in] state  State of the multibody system
     * @param[in] nu     Dimension of the control vector
     */
    ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu);

    /**
     * @brief Initialize the state residual model
     *
     * The default state reference is obtained from `StateAbstractTpl::zero()`, and `nu` from
     * `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state       State of the multibody system
     * @param[in] activation  Activation model
     */
    ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state);
    virtual ~ResidualFlyStateTpl();

    /**
     * @brief Compute the state residual
     *
     * @param[in] data  State residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the Jacobians of the state residual
     *
     * @param[in] data  State residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Return the reference state
     */
    const VectorXs &get_reference() const;

    /**
     * @brief Modify the reference state
     */
    void set_reference(const VectorXs &reference);

    /**
     * @brief Print relevant information of the state residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;

  private:
    VectorXs xref_; //!< Reference state
  };

} // namespace crocoddyl

namespace crocoddyl
{

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref, const std::size_t nu)
      : Base(state, 2, nu, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const VectorXs &xref)
      : Base(state, 2, false, false, false, false, true), xref_(xref)
  {
    if (static_cast<std::size_t>(xref_.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
  }

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   const std::size_t nu)
      : Base(state, 2, nu, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::ResidualFlyStateTpl(boost::shared_ptr<typename Base::StateAbstract> state)
      : Base(state, 2, false, false, false, false, true), xref_(state->zero()) {}

  template <typename Scalar>
  ResidualFlyStateTpl<Scalar>::~ResidualFlyStateTpl() {}

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                         const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->diff1(xref_, x, data->r); //diff1
    data->r.setZero();
    data->r.head(1) = x.tail(6).head(1);
    data->r.tail(1) = x.tail(2).head(1);
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                             const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // state_->Jdiff1(xref_, x, data->Rx, data->Rx, second);//diff1

    data->Rx.setZero();
    data->Rx.bottomRightCorner(2, 6).topLeftCorner(1, 1).diagonal().array() = (Scalar)1;
    data->Rx.bottomRightCorner(2, 2).bottomLeftCorner(1, 1).diagonal().array() = (Scalar)1;
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::print(std::ostream &os) const
  {
    os << "ResidualFlyState";
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &ResidualFlyStateTpl<Scalar>::get_reference() const
  {
    return xref_;
  }

  template <typename Scalar>
  void ResidualFlyStateTpl<Scalar>::set_reference(const VectorXs &reference)
  {
    xref_ = reference;
  }
} // namespace crocoddyl

namespace crocoddyl
{

  /**
   * @brief Frame translation residual
   *
   * This residual function defines the tracking of a frame translation as \f$\mathbf{r}=\mathbf{t}-\mathbf{t}^*\f$,
   * where \f$\mathbf{t},\mathbf{t}^*\in~\mathbb{R}^3\f$ are the current and reference frame translations, respectively.
   * Note that the dimension of the residual vector is 3. Furthermore, the Jacobians of the residual function are
   * computed analytically.
   *
   * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
   * `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  template <typename _Scalar>
  class ResidualKinoFrameTranslationTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataFrameTranslationTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::Vector3s Vector3s;

    /**
     * @brief Initialize the frame translation residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] id     Reference frame id
     * @param[in] xref   Reference frame translation
     * @param[in] nu     Dimension of the control vector
     */
    ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state, const pinocchio::FrameIndex,
                                    const Vector3s &xref, const std::size_t nu);

    /**
     * @brief Initialize the frame translation residual model
     *
     * The default `nu` is equals to StateAbstractTpl::get_nv().
     *
     * @param[in] state  State of the multibody system
     * @param[in] id     Reference frame id
     * @param[in] xref   Reference frame translation
     */
    ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state, const pinocchio::FrameIndex id,
                                    const Vector3s &xref);
    virtual ~ResidualKinoFrameTranslationTpl();

    /**
     * @brief Compute the frame translation residual
     *
     * @param[in] data  Frame translation residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the frame translation residual
     *
     * @param[in] data  Frame translation residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Create the frame translation residual data
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

    /**
     * @brief Return the reference frame id
     */
    pinocchio::FrameIndex get_id() const;

    /**
     * @brief Return the reference frame translation
     */
    const Vector3s &get_reference() const;

    /**
     * @brief Modify the reference frame id
     */
    void set_id(const pinocchio::FrameIndex id);

    /**
     * @brief Modify the reference frame translation reference
     */
    void set_reference(const Vector3s &reference);

    /**
     * @brief Print relevant information of the frame-translation residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;
    using Base::v_dependent_;

  private:
    pinocchio::FrameIndex id_;                                               //!< Reference frame id
    Vector3s xref_;                                                          //!< Reference frame translation
    boost::shared_ptr<typename StateKinodynamic::PinocchioModel> pin_model_; //!< Pinocchio model
  };

} // namespace crocoddyl

namespace crocoddyl
{

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s &xref, const std::size_t nu)
      : Base(state, 3, nu, true, false, false, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s &xref)
      : Base(state, 3, true, false, false, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::~ResidualKinoFrameTranslationTpl() {}

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                     const Eigen::Ref<const VectorXs> &x,
                                                     const Eigen::Ref<const VectorXs> &u)
  {
    // Compute the frame translation w.r.t. the reference frame
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // pinocchio::forwardKinematics(*pin_model_.get(), *d->pinocchio, q);
    pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
    data->r = d->pinocchio->oMf[id_].translation() - xref_;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                         const Eigen::Ref<const VectorXs> &x,
                                                         const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());

    // Compute the derivatives of the frame translation
    const std::size_t nv = state_->get_nv();
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // pinocchio::computeJointJacobians(*pin_model_.get(), *d->pinocchio, q);
    pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::WORLD, d->fJf);
    d->Rx.leftCols(nv).noalias() = d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualKinoFrameTranslationTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualKinoFrameTranslation {frame=" << pin_model_->frames[id_].name
       << ", tref=" << xref_.transpose().format(fmt) << "}";
  }

  template <typename Scalar>
  pinocchio::FrameIndex ResidualKinoFrameTranslationTpl<Scalar>::get_id() const
  {
    return id_;
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::Vector3s &ResidualKinoFrameTranslationTpl<Scalar>::get_reference() const
  {
    return xref_;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::set_id(const pinocchio::FrameIndex id)
  {
    id_ = id;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::set_reference(const Vector3s &translation)
  {
    xref_ = translation;
  }

} // namespace crocoddyl

namespace crocoddyl
{

  /**
   * @brief Frame placement residual
   *
   * This residual function defines the frame placement tracking as \f$\mathbf{r}=\mathbf{p}\ominus\mathbf{p}^*\f$, where
   * \f$\mathbf{p},\mathbf{p}^*\in~\mathbb{SE(3)}\f$ are the current and reference frame placements, respectively. Note
   * that the dimension of the residual vector is 6. Furthermore, the Jacobians of the residual function are
   * computed analytically.
   *
   * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
   * `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */

  template <typename _Scalar>
  struct ResidualDataKinoFramePlacementTpl : public ResidualDataAbstractTpl<_Scalar>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Matrix6xs Matrix6xs;
    typedef typename MathBase::Matrix6s Matrix6s;
    typedef typename MathBase::Vector6s Vector6s;

    template <template <typename Scalar> class Model>
    ResidualDataKinoFramePlacementTpl(Model<Scalar> *const model, DataCollectorAbstract *const data)
        : Base(model, data), rJf(6, 6), fJf(6, model->get_state()->get_nv())
    {
      r.setZero();
      rJf.setZero();
      fJf.setZero();
      // Check that proper shared data has been passed
      DataCollectorMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorMultibodyTpl<Scalar> *>(shared);
      if (d == NULL)
      {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
      }

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
    }

    pinocchio::DataTpl<Scalar> *pinocchio; //!< Pinocchio data
    pinocchio::SE3Tpl<Scalar> rMf;         //!< Error frame placement of the frame
    Matrix6s rJf;                          //!< Error Jacobian of the frame
    Matrix6xs fJf;                         //!< Local Jacobian of the frame

    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

  template <typename _Scalar>
  class ResidualKinoFramePlacementTpl : public ResidualModelAbstractTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef ResidualDataKinoFramePlacementTpl<Scalar> Data;
    typedef StateKinodynamicTpl<Scalar> StateKinodynamic;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef pinocchio::SE3Tpl<Scalar> SE3;

    /**
     * @brief Initialize the frame placement residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] id     Reference frame id
     * @param[in] pref   Reference frame placement
     * @param[in] nu     Dimension of the control vector
     */
    ResidualKinoFramePlacementTpl(boost::shared_ptr<StateKinodynamic> state, const pinocchio::FrameIndex id,
                                  const SE3 &pref, const std::size_t nu);

    /**
     * @brief Initialize the frame placement residual model
     *
     * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
     *
     * @param[in] state  State of the multibody system
     * @param[in] id     Reference frame id
     * @param[in] pref   Reference frame placement
     */
    ResidualKinoFramePlacementTpl(boost::shared_ptr<StateKinodynamic> state, const pinocchio::FrameIndex id,
                                  const SE3 &pref);
    virtual ~ResidualKinoFramePlacementTpl();

    /**
     * @brief Compute the frame placement residual
     *
     * @param[in] data  Frame placement residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Compute the derivatives of the frame placement residual
     *
     * @param[in] data  Frame-placement residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u);

    /**
     * @brief Create the frame placement residual data
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

    /**
     * @brief Return the reference frame id
     */
    pinocchio::FrameIndex get_id() const;

    /**
     * @brief Return the reference frame placement
     */
    const SE3 &get_reference() const;

    /**
     * @brief Modify the reference frame id
     */
    void set_id(const pinocchio::FrameIndex id);

    /**
     * @brief Modify the reference frame placement
     */
    void set_reference(const SE3 &reference);

    /**
     * @brief Print relevant information of the frame-placement residual
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream &os) const;

  protected:
    using Base::nu_;
    using Base::state_;
    using Base::u_dependent_;
    using Base::unone_;
    using Base::v_dependent_;

  private:
    pinocchio::FrameIndex id_;                                               //!< Reference frame id
    SE3 pref_;                                                               //!< Reference frame placement
    pinocchio::SE3Tpl<Scalar> oMf_inv_;                                      //!< Inverse reference placement
    boost::shared_ptr<typename StateKinodynamic::PinocchioModel> pin_model_; //!< Pinocchio model
  };
} // namespace crocoddyl

namespace crocoddyl
{

  template <typename Scalar>
  ResidualKinoFramePlacementTpl<Scalar>::ResidualKinoFramePlacementTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                       const pinocchio::FrameIndex id, const SE3 &pref,
                                                                       const std::size_t nu)
      : Base(state, 6, nu, true, false, false, false, false),
        id_(id),
        pref_(pref),
        oMf_inv_(pref.inverse()),
        pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFramePlacementTpl<Scalar>::ResidualKinoFramePlacementTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                       const pinocchio::FrameIndex id, const SE3 &pref)
      : Base(state, 6, true, false, false, false, false),
        id_(id),
        pref_(pref),
        oMf_inv_(pref.inverse()),
        pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFramePlacementTpl<Scalar>::~ResidualKinoFramePlacementTpl() {}

  template <typename Scalar>
  void ResidualKinoFramePlacementTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                   const Eigen::Ref<const VectorXs> &,
                                                   const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());

    // Compute the frame placement w.r.t. the reference frame
    pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
    d->rMf = oMf_inv_ * d->pinocchio->oMf[id_];
    data->r = pinocchio::log6(d->rMf).toVector();
  }

  template <typename Scalar>
  void ResidualKinoFramePlacementTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                       const Eigen::Ref<const VectorXs> &,
                                                       const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());

    // Compute the derivatives of the frame placement
    const std::size_t nv = state_->get_nv();
    pinocchio::Jlog6(d->rMf, d->rJf);
    pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
    data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf;
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualKinoFramePlacementTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualKinoFramePlacementTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    typename SE3::Quaternion qref;
    pinocchio::quaternion::assignQuaternion(qref, pref_.rotation());
    os << "ResidualKinoFramePlacement {frame=" << pin_model_->frames[id_].name
       << ", tref=" << pref_.translation().transpose().format(fmt) << ", qref=" << qref.coeffs().transpose().format(fmt)
       << "}";
  }

  template <typename Scalar>
  pinocchio::FrameIndex ResidualKinoFramePlacementTpl<Scalar>::get_id() const
  {
    return id_;
  }

  template <typename Scalar>
  const pinocchio::SE3Tpl<Scalar> &ResidualKinoFramePlacementTpl<Scalar>::get_reference() const
  {
    return pref_;
  }

  template <typename Scalar>
  void ResidualKinoFramePlacementTpl<Scalar>::set_id(const pinocchio::FrameIndex id)
  {
    id_ = id;
  }
  template <typename Scalar>
  void ResidualKinoFramePlacementTpl<Scalar>::set_reference(const SE3 &placement)
  {
    pref_ = placement;
    oMf_inv_ = placement.inverse();
  }

} // namespace crocoddyl

typedef crocoddyl::StateKinodynamicTpl<double> StateKinodynamic;
typedef crocoddyl::ActuationModelFloatingKinoBaseTpl<double> ActuationModelFloatingKinoBase;
typedef crocoddyl::DifferentialActionModelKinoDynamicsTpl<double> DifferentialActionModelContactKinoDynamics;
typedef crocoddyl::DifferentialActionDataKinoDynamicsTpl<double> DifferentialActionDataKinoDynamics;
typedef crocoddyl::ActivationModelWeightedQuadTpl<double> ActivationModelWeightedQuad;
typedef crocoddyl::ResidualFlyStateTpl<double> ResidualFlyState;
typedef crocoddyl::ResidualModelCentroidalAngularMomentumTpl<double> ResidualModelCentroidalAngularMomentum;
typedef crocoddyl::ActivationModelQuadraticBarrierTpl<double> ActivationModelQuadraticBarrier;
typedef crocoddyl::ActivationBoundsTpl<double> ActivationBounds;
typedef crocoddyl::ResidualModelCoMKinoPositionTpl<double> ResidualModelCoMKinoPosition;
typedef crocoddyl::ResidualKinoFrameTranslationTpl<double> ResidualKinoFrameTranslation;
typedef crocoddyl::ResidualKinoFramePlacementTpl<double> ResidualKinoFramePlacement;
typedef crocoddyl::MathBaseTpl<double> MathBase;
typename MathBase::VectorXs VectorXs;

std::vector<boost::shared_ptr<StateKinodynamic>> state_vector;
std::vector<boost::shared_ptr<ActuationModelFloatingKinoBase>> actuation_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> runningCostModel_vector;
std::vector<boost::shared_ptr<DifferentialActionModelContactKinoDynamics>> runningDAM_vector;
std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> runningModelWithRK4_vector;
std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract>> runningModelWithRK4_data;
// std::vector<boost::shared_ptr<ResidualKinoFrameTranslation>> residual_FrameRF;
// std::vector<boost::shared_ptr<ResidualKinoFrameTranslation>> residual_FrameLF;
std::vector<boost::shared_ptr<ResidualKinoFramePlacement>> residual_FrameRF;
std::vector<boost::shared_ptr<ResidualKinoFramePlacement>> residual_FrameLF;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> xRegCost_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> uRegCost_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> stateBoundCost_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> camBoundCost_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> comBoundCost_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> foot_trackR;
std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract>> foot_trackL;
// std::vector<Eigen::VectorXd> rf_foot_pos_vector;
// std::vector<Eigen::VectorXd> lf_foot_pos_vector;
std::vector<pinocchio::SE3> rf_foot_pos_vector;
std::vector<pinocchio::SE3> lf_foot_pos_vector;

std::vector<boost::shared_ptr<ActivationModelQuadraticBarrier>> state_activations;
std::vector<boost::shared_ptr<ActivationModelQuadraticBarrier>> state_activations2;
std::vector<boost::shared_ptr<ActivationModelQuadraticBarrier>> state_activations3;
std::vector<ActivationBounds> state_bounds;
std::vector<ActivationBounds> state_bounds2;
std::vector<ActivationBounds> state_bounds3;
std::vector<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>> runningDAM_data;

boost::shared_ptr<StateKinodynamic> state;
boost::shared_ptr<ActuationModelFloatingKinoBase> actuation;

Eigen::VectorXd traj_, u_traj_, weight_quad, weight_quad_u, weight_quad_zmp, weight_quad_cam, weight_quad_com, weight_quad_rf, weight_quad_lf;
double weight_quad_zmpx, weight_quad_zmpy, weight_quad_comx, weight_quad_comy, weight_quad_comz, weight_quad_camx, weight_quad_camy, weight_quad_rfx, weight_quad_rfy, weight_quad_rfz, weight_quad_lfx, weight_quad_lfy, weight_quad_lfz, weight_quad_rfroll, weight_quad_lfroll, weight_quad_rfpitch, weight_quad_lfpitch, weight_quad_rfyaw, weight_quad_lfyaw;
Eigen::MatrixXd lb_, ub_, lb_2, ub_2, lb_3, ub_3;

boost::shared_ptr<DifferentialActionModelContactKinoDynamics> terminalDAM;
boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel;
boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel;
Eigen::VectorXd x0;
Eigen::VectorXd u0;
boost::shared_ptr<crocoddyl::ShootingProblem> problemWithRK4;
std::vector<Eigen::VectorXd> xs, us;
std::vector<Eigen::VectorXd> xs_save, us_save;
std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;

double dt_;
pinocchio::Data data4;

Eigen::MatrixXd jointtick_, jointdottick_, xstatetick_, utick_, ustatetick_;

unsigned int N = 10; // number of nodes
unsigned int T = 1;  // number of trials
unsigned int MAXITER = 100;

int css_count = 0;


const std::string FILE_NAMES[2] =
        {
            ///change this directory when you use this code on the other computer///
            "/home/jhk/data/mpc/5_tocabi_.txt",
            "/home/jhk/data/mpc/6_tocabi_.txt",
        };

int main(int argc, char **argv)
{
  std::fstream file[2];
  for (int i = 0; i < 2; i++)
  {
    file[i].open(FILE_NAMES[i].c_str(), std::ios_base::out);
    file[i].precision(10);
  }
  Eigen::VectorQd q_;
  Eigen::VectorQd qdot, qddot, qdot_, qddot_;
  Eigen::VectorXd q_1, qdot1, qddot1, qdot_1, qddot_1;

  std::fstream read_file("/home/jhk/data/mpc/4_tocabi_.txt");
  std::string string_test;
  double jointvalue;
  int ticktest, squencetest;
  std::vector<int> walkingtick;
  std::vector<double> jointtick, jointdottick, xstatetick, utick, ustatetick;
  int c;
  if (read_file.is_open())
  {
    while (!read_file.eof())
    {
      read_file >> string_test;
      if (string_test == "walking_tick")
      {
        read_file >> ticktest;
        walkingtick.push_back(ticktest);
      }
      if (string_test == "q")
      {
        read_file >> squencetest;
        for (int i = 0; i < 19; i++)
        {
          read_file >> string_test;
          string_test.erase(find(string_test.begin(), string_test.end(), ','));
          jointvalue = atof(string_test.c_str());
          jointtick.push_back(jointvalue);
        }
      }
      if (string_test == "qdot")
      {
        read_file >> squencetest;
        for (int i = 0; i < 18; i++)
        {
          read_file >> string_test;
          string_test.erase(find(string_test.begin(), string_test.end(), ','));
          jointvalue = atof(string_test.c_str());
          jointdottick.push_back(jointvalue);
        }
      }
      if (string_test == "x_state")
      {
        read_file >> squencetest;
        for (int i = 0; i < 8; i++)
        {
          read_file >> string_test;
          string_test.erase(find(string_test.begin(), string_test.end(), ','));
          jointvalue = atof(string_test.c_str());
          xstatetick.push_back(jointvalue);
        }
      }
      if (string_test == "u")
      {
        read_file >> squencetest;
        for (int i = 0; i < 18; i++)
        {
          read_file >> string_test;
          string_test.erase(find(string_test.begin(), string_test.end(), ','));
          jointvalue = atof(string_test.c_str());
          utick.push_back(jointvalue);
        }
      }
      if (string_test == "ustate")
      {
        for (int i = 0; i < 4; i++)
        {
          read_file >> string_test;
          string_test.erase(find(string_test.begin(), string_test.end(), ','));
          jointvalue = atof(string_test.c_str());
          ustatetick.push_back(jointvalue);
        }
      }
      c++;
    }
  }
  std::vector<double> zmpxU, zmpxL, zmpyU, zmpyL, LFx, LFy, LFz, RFx, RFy, RFz;
  std::fstream read_file1("/home/jhk/data/mpc/3_tocabi_.txt");
  int a = 0;
  if (read_file1.is_open())
  {
    while (!read_file1.eof())
    {
      for (int i = 0; i < 30; i++)
      {
        read_file1 >> string_test;
        read_file1 >> string_test;
        read_file1 >> string_test;
        read_file1 >> string_test;
        string_test.erase(find(string_test.begin(), string_test.end(), 'u'));
        string_test.erase(find(string_test.begin(), string_test.end(), 'b'));

        zmpxL.push_back(atof(string_test.c_str()));

        read_file1 >> string_test;

        zmpxU.push_back(atof(string_test.c_str()));

        read_file1 >> string_test;

        read_file1 >> string_test;
        string_test.erase(find(string_test.begin(), string_test.end(), 'u'));
        string_test.erase(find(string_test.begin(), string_test.end(), 'b'));
        zmpyL.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        zmpyU.push_back(atof(string_test.c_str()));

        read_file1 >> string_test;
        RFx.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        RFy.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        RFz.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        LFx.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        LFy.push_back(atof(string_test.c_str()));
        read_file1 >> string_test;
        LFz.push_back(atof(string_test.c_str()));
      }
      a = a + 1;
     // std::cout << "a : " << a <<" " << walkingtick.size() <<std::endl;
      if (a == 234)
      {
        break;
      }

    }
  }


  jointtick_.resize(walkingtick.size() * 30, 19);
  jointdottick_.resize(walkingtick.size() * 30, 18);
  xstatetick_.resize(walkingtick.size() * 30, 8);
  utick_.resize(walkingtick.size() * 29, 18);
  ustatetick_.resize(walkingtick.size() * 29, 4);
  Eigen::MatrixXd zmpxU_, zmpxL_, zmpyU_, zmpyL_, LFx_, LFy_, LFz_, RFx_, RFy_, RFz_;
  zmpxU_.resize(a, 30);
  zmpxL_.resize(a, 30);
  zmpyU_.resize(a, 30);
  zmpyL_.resize(a, 30);
  LFx_.resize(a, 30);
  LFy_.resize(a, 30);
  LFz_.resize(a, 30);
  RFx_.resize(a, 30);
  RFy_.resize(a, 30);
  RFz_.resize(a, 30);

  for (int i = 0; i < walkingtick.size() * 30; i++)
  {
    for (int j = 0; j < 19; j++)
      jointtick_.row(i)(j) = jointtick[19 * i + j];

    for (int j = 0; j < 18; j++)
      jointdottick_.row(i)(j) = jointdottick[18 * i + j];
    for (int j = 0; j < 8; j++)
      xstatetick_.row(i)(j) = xstatetick[8 * i + j];
  }

  for (int i = 0; i < walkingtick.size() * 29; i++)
  {
    for (int j = 0; j < 18; j++)
      utick_.row(i)(j) = utick[18 * i + j];

    for (int j = 0; j < 4; j++)
      ustatetick_.row(i)(j) = ustatetick[4 * i + j];
  }

  for (int i = 0; i < a; i++)
  {
    for (int j = 0; j < 30; j++)
    {
      zmpxU_.row(i)(j) = zmpxU[30 * i + j];
      zmpxL_.row(i)(j) = zmpxL[30 * i + j];
      zmpyU_.row(i)(j) = zmpyU[30 * i + j];
      zmpyL_.row(i)(j) = zmpyL[30 * i + j];
      LFx_.row(i)(j) = LFx[30 * i + j];
      LFy_.row(i)(j) = LFy[30 * i + j];
      LFz_.row(i)(j) = LFz[30 * i + j];
      RFx_.row(i)(j) = RFx[30 * i + j];
      RFy_.row(i)(j) = RFy[30 * i + j];
      RFz_.row(i)(j) = RFz[30 * i + j];
    }
  }

  ros::init(argc, argv, "dyros_practice_1");
  ros::NodeHandle nh;
  std::string urdf_link = "/home/jhk/catkin_ws/src/dyros_tocabi_v2/tocabi_description/robots/dyros_tocabi_with_redhands.urdf";

  pinocchio::urdf::buildModel(urdf_link, model);
  pinocchio::Data data(model);
  model_data = data;
  q_ = randomConfiguration(model);
  qdot = Eigen::VectorXd::Zero(model.nv);
  qddot = Eigen::VectorXd::Zero(model.nv);
  qdot_ = Eigen::VectorXd::Zero(model.nv);
  qddot_ = Eigen::VectorXd::Zero(model.nv);

  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::Model model2;

  pinocchio::SE3 frame_pos;
  pinocchio::urdf::buildModel(urdf_link, root_joint, model2);
  pinocchio::JointIndex RFjoint_id = model2.getJointId("R_AnkleRoll_Joint");
  pinocchio::JointIndex LFjoint_id = model2.getJointId("L_AnkleRoll_Joint");
  int LFframe_id = model2.getFrameId("L_Foot_Link");
  int RFframe_id = model2.getFrameId("R_Foot_Link");

  Eigen::Matrix3d I3;
  I3.setIdentity();

  frame_pos.rotation() = I3;
  frame_pos.translation() << 0.03, 0, -0.1585;
  model2.addBodyFrame("LF_contact", LFjoint_id, frame_pos, LFframe_id);
  frame_pos.translation() << 0.03, 0, -0.1585;
  model2.addBodyFrame("RF_contact", RFjoint_id, frame_pos, RFframe_id);
  pinocchio::Data data1(model2);

  model1 = model2;
  model_data1 = data1;

  q_1 = Eigen::VectorXd::Zero(model1.nq);
  qdot1 = Eigen::VectorXd::Zero(model1.nv);
  qddot1 = Eigen::VectorXd::Zero(model1.nv);
  qdot_1 = Eigen::VectorXd::Zero(model1.nv);
  qddot_1 = Eigen::VectorXd::Zero(model1.nv);

  N = 30; // number of nodes
  T = 1;  // number of trials
  MAXITER = 300;

  // DifferentialActionDataKinoDynamicss
  pinocchio::Model model3;
  pinocchio::urdf::buildModel("/home/jhk/catkin_ws/src/tocabi_cc/robots/dyros_tocabi_with_redhands.urdf",
                              pinocchio::JointModelFreeFlyer(), model3);
  pinocchio::Data data3(model3);
  Eigen::VectorXd q_3;
  q_3 = randomConfiguration(model3);
  q_3 << 0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0;
  pinocchio::forwardKinematics(model3, data3, q_3);
  pinocchio::centerOfMass(model3, data3, q_3, false);
  pinocchio::updateFramePlacements(model3, data3);
  model3.lowerPositionLimit.head<7>().array() = -1;
  model3.upperPositionLimit.head<7>().array() = 1.;
  std::cout << "Rf " << std::endl;
  /// std::cout << data3.oMi[RFjoint_id].translation << std::endl;
  const pinocchio::SE3 rf_foot_pos0 = data3.oMf[RFframe_id];
  Eigen::Vector3d rf_foot_pos = rf_foot_pos0.translation();
  const pinocchio::SE3 lf_foot_pos0 = data3.oMf[LFframe_id];
  Eigen::Vector3d lf_foot_pos = lf_foot_pos0.translation();

  std::cout << rf_foot_pos0.translation() << std::endl;
  // const pinocchio::SE3::Vector3& lf_foot_pos0 = data3.oMi[LFjoint_id].translation;

  const std::string RF = "R_AnkleRoll_Joint";
  const std::string LF = "L_AnkleRoll_Joint";

  state =
      boost::make_shared<StateKinodynamic>(boost::make_shared<pinocchio::Model>(model3));
  actuation =
      boost::make_shared<ActuationModelFloatingKinoBase>(state);

  traj_.resize(45);
  traj_.setZero();
  traj_.head(19) << 0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0;
  traj_(37) = 0.08;
  traj_(38) = 0.0;
  traj_(39) = 0.0;
  traj_(40) = 0.00;

  u_traj_.resize(22);
  u_traj_.setZero();
  u_traj_(18) = 1.0;

  lb_.resize(2, N);
  lb_.setOnes();
  lb_ = -3.2 * lb_;

  ub_.resize(2, N);
  ub_.setOnes();
  ub_ = 3.2 * ub_;
  // N-4가 하고싶으면 N-5부터?
  for (int i = 0; i < N - 5; i++)
  {
    lb_(0, i) = 0.0;
    ub_(0, i) = 0.2;
  }

  for (int i = N - 5; i < N; i++)
  {
    lb_(0, i) = 0.15;
    ub_(0, i) = 0.4;
  }

  for (int i = 0; i < N - 4; i++)
  {
    lb_(1, i) = -0.2;
    ub_(1, i) = 0.2;
  }

  for (int i = N - 4; i < N; i++)
  {
    lb_(1, i) = 0.05;
    ub_(1, i) = 0.2;
  }

  lb_2.resize(2, N);
  lb_2.setZero();

  ub_2.resize(2, N);
  ub_2.setZero();

  lb_3.resize(3, N);
  lb_3.setZero();

  ub_3.resize(3, N);
  ub_3.setZero();

  nh.getParam("/dyros_practice_1/weight_quad_zmpx", weight_quad_zmpx);
  nh.getParam("/dyros_practice_1/weight_quad_zmpy", weight_quad_zmpy);
  nh.getParam("/dyros_practice_1/weight_quad_camx", weight_quad_camx);
  nh.getParam("/dyros_practice_1/weight_quad_camy", weight_quad_camy);
  nh.getParam("/dyros_practice_1/weight_quad_comx", weight_quad_comx);
  nh.getParam("/dyros_practice_1/weight_quad_comy", weight_quad_comy);
  nh.getParam("/dyros_practice_1/weight_quad_comz", weight_quad_comz);

  nh.getParam("/dyros_practice_1/weight_quad_rfx", weight_quad_rfx);
  nh.getParam("/dyros_practice_1/weight_quad_rfy", weight_quad_rfy);
  nh.getParam("/dyros_practice_1/weight_quad_rfz", weight_quad_rfz);

  nh.getParam("/dyros_practice_1/weight_quad_lfx", weight_quad_lfx);
  nh.getParam("/dyros_practice_1/weight_quad_lfy", weight_quad_lfy);
  nh.getParam("/dyros_practice_1/weight_quad_lfz", weight_quad_lfz);

  nh.getParam("/dyros_practice_1/weight_quad_rfroll", weight_quad_rfroll);
  nh.getParam("/dyros_practice_1/weight_quad_rfpitch", weight_quad_rfpitch);
  nh.getParam("/dyros_practice_1/weight_quad_rfyaw", weight_quad_rfyaw);

  nh.getParam("/dyros_practice_1/weight_quad_lfroll", weight_quad_lfroll);
  nh.getParam("/dyros_practice_1/weight_quad_lfpitch", weight_quad_lfpitch);
  nh.getParam("/dyros_practice_1/weight_quad_lfyaw", weight_quad_lfyaw);

  weight_quad.resize(state->get_ndx());
  weight_quad.setZero();
  weight_quad(36) = 10;

  weight_quad_u.resize(22);
  weight_quad_u.setZero();
  weight_quad_u(18) = 1.0;

  weight_quad_zmp.resize(2);
  weight_quad_cam.resize(2);
  weight_quad_com.resize(3);
  weight_quad_rf.resize(6);
  weight_quad_lf.resize(6);

  weight_quad_zmp.setOnes();
  weight_quad_cam.setOnes();
  weight_quad_com.setOnes();
  weight_quad_rf.setOnes();
  weight_quad_lf.setOnes();

  weight_quad_zmp << weight_quad_zmpx, weight_quad_zmpy;
  weight_quad_cam << weight_quad_camy, weight_quad_camx;
  weight_quad_com << weight_quad_comx, weight_quad_comy, weight_quad_comz;
  weight_quad_rf << weight_quad_rfx, weight_quad_rfy, weight_quad_rfz, weight_quad_rfroll, weight_quad_rfpitch, weight_quad_rfyaw;
  weight_quad_lf << weight_quad_lfx, weight_quad_lfy, weight_quad_lfz, weight_quad_lfroll, weight_quad_lfpitch, weight_quad_lfyaw;

  dt_ = 1.2 / double(N);

  for (int i = 0; i < N; i++)
  {
    state_vector.push_back(boost::make_shared<StateKinodynamic>(boost::make_shared<pinocchio::Model>(model3)));
    state_bounds.push_back(ActivationBounds(lb_.col(i), ub_.col(i)));
    state_bounds2.push_back(ActivationBounds(lb_2.col(i), ub_2.col(i)));
    state_bounds3.push_back(ActivationBounds(lb_3.col(i), ub_3.col(i)));
    state_activations.push_back(boost::make_shared<ActivationModelQuadraticBarrier>(state_bounds[i]));
    state_activations2.push_back(boost::make_shared<ActivationModelQuadraticBarrier>(state_bounds2[i]));
    state_activations3.push_back(boost::make_shared<ActivationModelQuadraticBarrier>(state_bounds3[i]));
    actuation_vector.push_back(boost::make_shared<ActuationModelFloatingKinoBase>(state_vector[i]));
    xRegCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state, boost::make_shared<ActivationModelWeightedQuad>(weight_quad), boost::make_shared<crocoddyl::ResidualModelState>(state_vector[i], traj_, actuation_vector[i]->get_nu() + 4)));
    uRegCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state, boost::make_shared<ActivationModelWeightedQuad>(weight_quad_u), boost::make_shared<crocoddyl::ResidualModelControl>(state_vector[i], u_traj_))); //, actuation_vector[i]->get_nu() + 1)));

    stateBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], state_activations[i], boost::make_shared<ResidualFlyState>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
    camBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], boost::make_shared<ActivationModelWeightedQuad>(weight_quad_cam), boost::make_shared<ResidualModelCentroidalAngularMomentum>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
    comBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], boost::make_shared<ActivationModelWeightedQuad>(weight_quad_com), boost::make_shared<ResidualModelCoMKinoPosition>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
    pinocchio::SE3 rf_foot_temp(Eigen::Matrix3d::Identity(), rf_foot_pos);
    pinocchio::SE3 lf_foot_temp(Eigen::Matrix3d::Identity(), lf_foot_pos);
    rf_foot_pos_vector.push_back(rf_foot_temp);
    lf_foot_pos_vector.push_back(lf_foot_temp);
    //  rf_foot_pos_vector.push_back(rf_foot_pos);
    //  lf_foot_pos_vector.push_back(lf_foot_pos);
    //    residual_FrameRF.push_back(boost::make_shared<ResidualKinoFrameTranslation>(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    //    residual_FrameLF.push_back(boost::make_shared<ResidualKinoFrameTranslation>(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    residual_FrameRF.push_back(boost::make_shared<ResidualKinoFramePlacement>(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    residual_FrameLF.push_back(boost::make_shared<ResidualKinoFramePlacement>(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    foot_trackR.push_back(boost::make_shared<crocoddyl::CostModelResidual>(state_vector[i], boost::make_shared<ActivationModelWeightedQuad>(weight_quad_rf), residual_FrameRF[i]));
    foot_trackL.push_back(boost::make_shared<crocoddyl::CostModelResidual>(state_vector[i], boost::make_shared<ActivationModelWeightedQuad>(weight_quad_lf), residual_FrameLF[i]));
  }

  std::cout << "state->get_nv()" << std::endl;
  std::cout << state->get_nv() << std::endl; /// nv_

  std::cout << "state->get_nq()" << std::endl;
  std::cout << state->get_nq() << std::endl; // nq_

  std::cout << "state->get_nx()" << std::endl;
  std::cout << state->get_nx() << std::endl; // nx_

  std::cout << "state->get_ndx()" << std::endl;
  std::cout << state->get_ndx() << std::endl; // ndx_

  std::cout << "actuation->get_nu()" << std::endl;
  std::cout << actuation->get_nu() << std::endl;

  for (int i = 0; i < N - 1; i++)
  {
    runningCostModel_vector.push_back(boost::make_shared<crocoddyl::CostModelSum>(state_vector[i], actuation_vector[i]->get_nu() + 4));
    runningCostModel_vector[i]->addCost("stateReg", stateBoundCost_vector[i], weight_quad_zmp(0));
    runningCostModel_vector[i]->addCost("camReg", camBoundCost_vector[i], 1e0);
    runningCostModel_vector[i]->addCost("comReg", comBoundCost_vector[i], 1e0);
    runningCostModel_vector[i]->addCost("footReg1", foot_trackR[i], 1e0);
    runningCostModel_vector[i]->addCost("footReg2", foot_trackL[i], 1e0);
  }

  terminalCostModel =
      boost::make_shared<crocoddyl::CostModelSum>(state_vector[N - 1], actuation_vector[N - 1]->get_nu() + 4);

  terminalCostModel->addCost("stateReg", stateBoundCost_vector[N - 1], weight_quad_zmp(0));
  terminalCostModel->addCost("camReg", camBoundCost_vector[N - 1], 1e0);
  terminalCostModel->addCost("comReg", comBoundCost_vector[N - 1], 1e0);
  terminalCostModel->addCost("footReg1", foot_trackR[N - 1], 1e0);
  terminalCostModel->addCost("footReg2", foot_trackL[N - 1], 1e0);

  terminalDAM =
      boost::make_shared<DifferentialActionModelContactKinoDynamics>(state_vector[N - 1], actuation_vector[N - 1],
                                                                     terminalCostModel);

  for (int i = 0; i < N - 1; i++)
  {
    runningDAM_vector.push_back(boost::make_shared<DifferentialActionModelContactKinoDynamics>(state_vector[i], actuation_vector[i],
                                                                                               runningCostModel_vector[i]));
    runningModelWithRK4_vector.push_back(boost::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM_vector[i], dt_));

    // runningModelWithRK4_vector.push_back(boost::make_shared<crocoddyl::IntegratedActionModelRK>(runningDAM_vector[i], crocoddyl::RKType::two, dt_));
  }

  x0.resize(state->get_nx() + 8);
  u0.resize(actuation->get_nu() + 4);
  u0.setZero();
  x0.setZero();
  x0.segment<19>(0) << 0, 0, 0.80783, 0, 0, 0, 1, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0, 0.0, 0.0, -0.55, 1.26, -0.71, 0.0;
  x0.tail(8)(0) = data3.com[0](0);
  x0.tail(8)(2) = data3.com[0](0);
  terminalModel =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(terminalDAM, dt_);
  problemWithRK4 =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModelWithRK4_vector, terminalModel);

  for (int i = 0; i < N - 1; i++)
  {
    runningModelWithRK4_data.push_back(runningModelWithRK4_vector[i]->createData());
    runningDAM_vector[i]->createData();
  }

  problemWithRK4->set_nthreads(6);
  std::cout << "thread " << problemWithRK4->get_nthreads() << std::endl; // " " << problemWithRK4->enableMultithreading()<< std::endl;

  crocoddyl::SolverBoxFDDP ddp(problemWithRK4);

  for (int i = 0; i < N; i++)
  {
    xs.push_back(x0);
    if (i != N - 1)
      us.push_back(u0);
  }

  bool CALLBACKS = false;
  if (CALLBACKS)
  {
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);
  }

  int css;
  T = 5;
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i)
  {
    crocoddyl::Timer timer;    
    css = ddp.solve(xs, us, MAXITER, 0.5);
    duration[i] = timer.get_duration();
    std::cout << "aftersolve" << std::endl;
    std::cout << "sss " << ddp.get_iter() << "  " << css << std::endl;

    xs = ddp.get_xs();
    us = ddp.get_us();

    std::cout << "css " << ddp.get_iter() << " " << ddp.get_is_feasible() << std::endl;

    double avrg_duration = duration[i];
    double min_duration = duration.minCoeff();
    double max_duration = duration.maxCoeff();

    std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
              << std::endl;
    /*
    for (int i = 0; i < N - 1; i++)
    {
      std::cout << "q " << i << std::endl;
      for (int j = 0; j < 19; j++)
      {
        std::cout << xs[i][j] << ", ";
      }
      std::cout << "qdot " << i << std::endl;
      {
        for (int j = 19; j < 37; j++)
        {
          std::cout << xs[i][j] << ", ";
        }
      }
      std::cout << "x_state " << i << std::endl;
      {
        for (int j = 37; j < 45; j++)
        {
          std::cout << xs[i][j] << ", ";
        }
      }
      std::cout << std::endl;
      std::cout << "u " << i << std::endl;
      for (int j = 0; j < actuation->get_nu(); j++)
      {
        std::cout << us[i][j] << ", ";
      }
      std::cout << "ustate" << std::endl;
      for (int j = actuation->get_nu(); j < actuation->get_nu() + 4; j++)
      {
        std::cout << us[i][j] << ", ";
      }
      std::cout << std::endl;
    }

    std::cout << "q " << N - 1 << std::endl;
    for (int j = 0; j < 19; j++)
    {
      std::cout << xs[N - 1][j] << ", ";
    }
    std::cout << "qdot " << N - 1 << std::endl;
    {
      for (int j = 19; j < 37; j++)
      {
        std::cout << xs[N - 1][j] << ", ";
      }
    }
    std::cout << "x_state " << N - 1 << std::endl;
    {
      for (int j = 37; j < 45; j++)
      {
        std::cout << xs[N - 1][j] << ", ";
      }
    }*/
  }

  int walking_ti = 0;
  std::cout << weight_quad_zmp << std::endl;
  css = 1;

  while (ros::ok())
  {
    int N_temp = 1;
    for (int i = 0; i < N; i++)
    {
      state_bounds[i].lb(0) = zmpxL_(walking_ti, N_temp * (i));
      state_bounds[i].ub(0) = zmpxU_(walking_ti, N_temp * (i));
      state_bounds[i].lb(1) = zmpyL_(walking_ti, N_temp * (i));
      state_bounds[i].ub(1) = zmpyU_(walking_ti, N_temp * (i));
      state_activations[i]->set_bounds(state_bounds[i]);

      rf_foot_pos_vector[i].translation() << RFx_(walking_ti, N_temp * (i)), RFy_(walking_ti, N_temp * (i)), RFz_(walking_ti, N_temp * (i));
      lf_foot_pos_vector[i].translation() << LFx_(walking_ti, N_temp * (i)), LFy_(walking_ti, N_temp * (i)), LFz_(walking_ti, N_temp * (i));
      residual_FrameRF[i]->set_reference(rf_foot_pos_vector[i]);
      residual_FrameLF[i]->set_reference(lf_foot_pos_vector[i]);

      file[0] << walking_ti << " " << i << " lb " << state_bounds[i].lb(0) << "ub " << state_bounds[i].ub(0) << " "
              << " lb " << state_bounds[i].lb(1) << "ub " << state_bounds[i].ub(1) << " " << residual_FrameRF[i]->get_reference().translation().transpose() << " " << residual_FrameLF[i]->get_reference().translation().transpose() << std::endl;
    }

    bool CALLBACKS = false;
    if (CALLBACKS)
    {
      ddp.setCallbacks(cbs);
    }
    
    problemWithRK4->set_x0(xs[1]);
    crocoddyl::SolverBoxFDDP ddp(problemWithRK4);
    //us[0].setZero();
    if(walking_ti == 7)
    {
      for(int j = 1; j < N; j++)
      {
       // std::cout << "j  " << j << std::endl;
       // std::cout << (xs[j].head(19) - jointtick_.row(30*(walking_ti-1) + j).transpose()).transpose() << std::endl; 
       // std::cout << (xs[j].head(37).tail(18)  - jointdottick_.row(30*(walking_ti-1) + j).transpose()).transpose() << std::endl;
       // std::cout << (xs[j].tail(8)  - xstatetick_.row(30*(walking_ti-1) + j).transpose()).transpose() << std::endl;

        
        //std::cout << jointtick_.row(30*(walking_ti-1) + j) << std::endl;
      
        xs[j].head(19) = jointtick_.row(30*(walking_ti) + j).transpose();
        xs[j].head(37).tail(18) = jointdottick_.row(30*(walking_ti) + j).transpose();
        xs[j].tail(8) = xstatetick_.row(30*(walking_ti) + j);
      }  

      for(int j = 0; j < N - 1; j++)
      { 
        std::cout << "j  " << j << std::endl;
        //std::cout << (us[j].head(18)  - utick_.row(29*(walking_ti - 1) + j).transpose()).transpose() << std::endl;
        //std::cout << (us[j].tail(4)  - ustatetick_.row(29*(walking_ti-1) + j).transpose()).transpose() << std::endl;
       // std::cout << us[j].head(18).transpose() << std::endl;
        std::cout << utick_.row(29*(walking_ti) + j) << std::endl;
       // std::cout << utick_.row(29*(walking_ti - 1) + j) << std::endl;
       
        us[j].head(18) = utick_.row(29*(walking_ti) + j).transpose();
        us[j].tail(4) = ustatetick_.row(29*(walking_ti) + j).transpose();
      }
    }
    
    T = 1;
    Eigen::ArrayXd duration(T);
    for (unsigned int i = 0; i < T; ++i)
    {
      crocoddyl::Timer timer;
      css = ddp.solve(xs, us, MAXITER, 0.5);
      duration[i] = timer.get_duration();
      std::cout << "aftersolve1" << std::endl;
      std::cout << "iter :  " << ddp.get_iter() << "  " << css << std::endl;

      if (css == 0 && ddp.get_iter() != MAXITER)
      {
        xs = xs_save;
        us = us_save;
        problemWithRK4->set_x0(xs[1]);
      }
      else
      {
        xs = ddp.get_xs();
        us = ddp.get_us();
      }

      std::cout << "css " << ddp.get_iter() << " " << ddp.get_is_feasible() << std::endl;

      double duration_dis;
      if (i == 0)
      {
        duration_dis = duration[i];
        xs_save = xs;
        us_save = us;
      }
      else if (duration_dis - duration[i] > 0)
      {
        duration_dis = duration[i];
        xs_save = xs;
        us_save = us;
      }

      double avrg_duration = duration[i]; //.sum() / T;
      double min_duration = duration.minCoeff();
      double max_duration = duration.maxCoeff();

      std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration << "-" << max_duration << ")"
                << std::endl;
    }

    if (css == 0)
    {
      css_count = css_count + 1;
    }
    std::cout << "walking_tick " << walking_ti << " css_count : " << css_count << std::endl;
    file[1] << "walking_tick " << walking_ti << " css " << ddp.get_iter() << " " << css << " " << ddp.get_is_feasible() << std::endl;
    for (int i = 0; i < N - 1; i++)
    {
      file[1] << "q " << i << std::endl;
      for (int j = 0; j < 19; j++)
      {
        file[1] << xs_save[i][j] << ", ";
      }
      file[1] << "qdot " << i << std::endl;
      {
        for (int j = 19; j < 37; j++)
        {
          file[1] << xs_save[i][j] << ", ";
        }
      }
      file[1] << "x_state " << i << std::endl;
      {
        for (int j = 37; j < 45; j++)
        {
          file[1] << xs_save[i][j] << ", ";
        }
      }
      file[1] << std::endl;
      file[1] << "u " << i << std::endl;
      for (int j = 0; j < actuation->get_nu(); j++)
      {
        file[1] << us_save[i][j] << ", ";
      }
      file[1] << "ustate" << std::endl;
      for (int j = actuation->get_nu(); j < actuation->get_nu() + 4; j++)
      {
        file[1] << us_save[i][j] << ", ";
      }
      file[1] << std::endl;
    }

    file[1] << "q " << N - 1 << std::endl;
    for (int j = 0; j < 19; j++)
    {
      file[1] << xs_save[N - 1][j] << ", ";
    }
    file[1] << "qdot " << N - 1 << std::endl;
    {
      for (int j = 19; j < 37; j++)
      {
        file[1] << xs_save[N - 1][j] << ", ";
      }
    }
    file[1] << "x_state " << N - 1 << std::endl;
    {
      for (int j = 37; j < 45; j++)
      {
        file[1] << xs_save[N - 1][j] << ", ";
      }
    }
    file[1] << std::endl;
    walking_ti = walking_ti + 1;
    if(walking_ti == 19)
    {
      break;
    }
  }
}
