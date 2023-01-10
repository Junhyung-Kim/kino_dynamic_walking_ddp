#include <pinocchio/fwd.hpp>
#include "ros/ros.h"
#include <fstream>
#include <string>
#include <vector>

#include "crocoddyl/multibody/actuations/kino-base.hpp"
#include "crocoddyl/multibody/residuals/kinostate.hpp"
#include "crocoddyl/multibody/residuals/kinoframe-placement.hpp"
#include "crocoddyl/multibody/residuals/kinoframe-translation.hpp"
#include "crocoddyl/multibody/residuals/com-kino-position.hpp"
#include "crocoddyl/multibody/residuals/centroidal-angular-momentum.hpp"

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
#include "crocoddyl/multibody/actions/kinodyn.hpp"
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

#include <string.h>
#include <torch/torch.h>

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

typedef crocoddyl::MathBaseTpl<double> MathBase;
typename MathBase::VectorXs VectorXs;

std::vector<boost::shared_ptr<crocoddyl::StateKinodynamic>> state_vector;
std::vector<boost::shared_ptr<crocoddyl::ActuationModelKinoBase>> actuation_vector;
std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> runningCostModel_vector;
std::vector<boost::shared_ptr<crocoddyl::DifferentialActionModelKinoDynamics>> runningDAM_vector;
std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> runningModelWithRK4_vector;
std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract>> runningModelWithRK4_data;
// std::vector<boost::shared_ptr<ResidualKinoFrameTranslation>> residual_FrameRF;
// std::vector<boost::shared_ptr<ResidualKinoFrameTranslation>> residual_FrameLF;
std::vector<boost::shared_ptr<crocoddyl::ResidualKinoFramePlacement>> residual_FrameRF;
std::vector<boost::shared_ptr<crocoddyl::ResidualKinoFramePlacement>> residual_FrameLF;
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

std::vector<boost::shared_ptr<crocoddyl::ActivationModelQuadraticBarrier>> state_activations;
std::vector<boost::shared_ptr<crocoddyl::ActivationModelQuadraticBarrier>> state_activations2;
std::vector<boost::shared_ptr<crocoddyl::ActivationModelQuadraticBarrier>> state_activations3;
std::vector<crocoddyl::ActivationBounds> state_bounds;
std::vector<crocoddyl::ActivationBounds> state_bounds2;
std::vector<crocoddyl::ActivationBounds> state_bounds3;
std::vector<boost::shared_ptr<crocoddyl::DifferentialActionDataAbstract>> runningDAM_data;

boost::shared_ptr<crocoddyl::StateKinodynamic> state;
boost::shared_ptr<crocoddyl::ActuationModelKinoBase> actuation;

Eigen::VectorXd traj_, u_traj_, weight_quad, weight_quad_u, weight_quad_zmp, weight_quad_cam, weight_quad_com, weight_quad_rf, weight_quad_lf;
double weight_quad_zmpx, weight_quad_zmpy, weight_quad_comx, weight_quad_comy, weight_quad_comz, weight_quad_camx, weight_quad_camy, weight_quad_rfx, weight_quad_rfy, weight_quad_rfz, weight_quad_lfx, weight_quad_lfy, weight_quad_lfz, weight_quad_rfroll, weight_quad_lfroll, weight_quad_rfpitch, weight_quad_lfpitch, weight_quad_rfyaw, weight_quad_lfyaw;
Eigen::MatrixXd lb_, ub_, lb_2, ub_2, lb_3, ub_3;

boost::shared_ptr<crocoddyl::DifferentialActionModelKinoDynamics> terminalDAM;
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
      boost::make_shared<crocoddyl::StateKinodynamic>(boost::make_shared<pinocchio::Model>(model3));
  actuation =
      boost::make_shared<crocoddyl::ActuationModelKinoBase>(state);

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

  nh.getParam("/dyros_practice/weight_quad_zmpx", weight_quad_zmpx);
  nh.getParam("/dyros_practice/weight_quad_zmpy", weight_quad_zmpy);
  nh.getParam("/dyros_practice/weight_quad_camx", weight_quad_camx);
  nh.getParam("/dyros_practice/weight_quad_camy", weight_quad_camy);
  nh.getParam("/dyros_practice/weight_quad_comx", weight_quad_comx);
  nh.getParam("/dyros_practice/weight_quad_comy", weight_quad_comy);
  nh.getParam("/dyros_practice/weight_quad_comz", weight_quad_comz);

  nh.getParam("/dyros_practice/weight_quad_rfx", weight_quad_rfx);
  nh.getParam("/dyros_practice/weight_quead_rfy", weight_quad_rfy);
  nh.getParam("/dyros_practice/weight_quad_rfz", weight_quad_rfz);

  nh.getParam("/dyros_practice/weight_quad_lfx", weight_quad_lfx);
  nh.getParam("/dyros_practice/weight_quad_lfy", weight_quad_lfy);
  nh.getParam("/dyros_practice/weight_quad_lfz", weight_quad_lfz);

  nh.getParam("/dyros_practice/weight_quad_rfroll", weight_quad_rfroll);
  nh.getParam("/dyros_practice/weight_quad_rfpitch", weight_quad_rfpitch);
  nh.getParam("/dyros_practice/weight_quad_rfyaw", weight_quad_rfyaw);

  nh.getParam("/dyros_practice/weight_quad_lfroll", weight_quad_lfroll);
  nh.getParam("/dyros_practice/weight_quad_lfpitch", weight_quad_lfpitch);
  nh.getParam("/dyros_practice/weight_quad_lfyaw", weight_quad_lfyaw);

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

  std::cout << weight_quad_lf<< std::endl;

  dt_ = 1.2 / double(N);

  for (int i = 0; i < N; i++)
  {
    state_vector.push_back(boost::make_shared<crocoddyl::StateKinodynamic>(boost::make_shared<pinocchio::Model>(model3)));
    state_bounds.push_back(crocoddyl::ActivationBounds(lb_.col(i), ub_.col(i)));
    state_bounds2.push_back(crocoddyl::ActivationBounds(lb_2.col(i), ub_2.col(i)));
    state_bounds3.push_back(crocoddyl::ActivationBounds(lb_3.col(i), ub_3.col(i)));
    state_activations.push_back(boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(state_bounds[i]));
    state_activations2.push_back(boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(state_bounds2[i]));
    state_activations3.push_back(boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(state_bounds3[i]));
    actuation_vector.push_back(boost::make_shared<crocoddyl::ActuationModelKinoBase>(state_vector[i]));
    xRegCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state, boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad), boost::make_shared<crocoddyl::ResidualModelState>(state_vector[i], traj_, actuation_vector[i]->get_nu() + 4)));
    uRegCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state, boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad_u), boost::make_shared<crocoddyl::ResidualModelControl>(state_vector[i], u_traj_))); //, actuation_vector[i]->get_nu() + 1)));

    stateBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], state_activations[i], boost::make_shared<crocoddyl::ResidualFlyState>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
    camBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad_cam), boost::make_shared<crocoddyl::ResidualModelCentroidalAngularMomentum>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
    comBoundCost_vector.push_back(boost::make_shared<crocoddyl::CostModelResidual>(
        state_vector[i], boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad_com), boost::make_shared<crocoddyl::ResidualModelCoMKinoPosition>(state_vector[i], actuation_vector[i]->get_nu() + 4)));
   
    pinocchio::SE3 rf_foot_temp(Eigen::Matrix3d::Identity(), rf_foot_pos);
    pinocchio::SE3 lf_foot_temp(Eigen::Matrix3d::Identity(), lf_foot_pos);
    rf_foot_pos_vector.push_back(rf_foot_temp);
    lf_foot_pos_vector.push_back(lf_foot_temp);
    //  rf_foot_pos_vector.push_back(rf_foot_pos);
    //  lf_foot_pos_vector.push_back(lf_foot_pos);
    //    residual_FrameRF.push_back(boost::make_shared<ResidualKinoFrameTranslation>(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    //    residual_FrameLF.push_back(boost::make_shared<ResidualKinoFrameTranslation>(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    residual_FrameRF.push_back(boost::make_shared<crocoddyl::ResidualKinoFramePlacement>(state_vector[i], RFframe_id, rf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    residual_FrameLF.push_back(boost::make_shared<crocoddyl::ResidualKinoFramePlacement>(state_vector[i], LFframe_id, lf_foot_pos_vector[i], actuation_vector[i]->get_nu() + 4));
    foot_trackR.push_back(boost::make_shared<crocoddyl::CostModelResidual>(state_vector[i], boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad_rf), residual_FrameRF[i]));
    foot_trackL.push_back(boost::make_shared<crocoddyl::CostModelResidual>(state_vector[i], boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weight_quad_lf), residual_FrameLF[i]));
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
      boost::make_shared<crocoddyl::DifferentialActionModelKinoDynamics>(state_vector[N - 1], actuation_vector[N - 1], terminalCostModel);

  for (int i = 0; i < N - 1; i++)
  {
    runningDAM_vector.push_back(boost::make_shared<crocoddyl::DifferentialActionModelKinoDynamics>(state_vector[i], actuation_vector[i],
                                                                                               runningCostModel_vector[i]));
    runningModelWithRK4_vector.push_back(boost::make_shared<crocoddyl::IntegratedActionModelEuler>(runningDAM_vector[i], dt_));

    // runningModelWithRK4_vector.push_back(boost::make_shared<crocoddyl::IntegratedActionModelRK>(runningDAM_vector[i], crocoddyl::RKType::two, dt_));
  }
  //std::cout << "Aaa" <<std::endl;
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
    //runningDAM_vector[i]->createData();
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
  T = 1;
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i)
  {
    crocoddyl::Timer timer;    
    css = ddp.solve(xs, us, MAXITER);
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
    }
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

    T = 1;
    Eigen::ArrayXd duration(T);
    MAXITER = 300;
    for (unsigned int i = 0; i < T; ++i)
    {
      crocoddyl::Timer timer;
      css = ddp.solve(xs, us, MAXITER);
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
    std::cout << "ddp residual"<< std::endl;

    if(walking_ti == 15)
    {
      break;
    }
  }
}

