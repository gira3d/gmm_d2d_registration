// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use the general purpose non-linear
    optimization routines from the dlib C++ Library.

    The library provides implementations of the conjugate gradient,  BFGS,
    L-BFGS, and BOBYQA optimization algorithms.  These algorithms allow you to
    find the minimum of a function of many input variables.  This example walks
    though a few of the ways you might put these routines to use.

*/


#include <dlib/optimization.h>
#include <iostream>
#include <Eigen/SVD>

#include <gmm/GMM3.h>

using namespace std;
using namespace dlib;

#define NUM_DIMS 3

// ----------------------------------------------------------------------------------------
// In dlib, the general purpose solvers optimize functions that take a column
// vector as input and return a double.  So here we make a typedef for a
// variable length column vector of doubles.  This is the type we will use to
// represent the input to our objective functions which we will be minimizing.
typedef matrix<float,0,1> column_vector;

typedef Eigen::Matrix3f array;
typedef Eigen::Vector3f vector;
typedef std::vector<Eigen::Matrix3f> array3D;
typedef std::vector<Eigen::Vector3f> vector3D;
typedef std::vector<std::vector<Eigen::Matrix3f>> array4D;
// ----------------------------------------------------------------------------------------
// Below we create a few functions.  When you get down into main() you will see that
// we can use the optimization algorithms to find the minimums of these functions.
// ----------------------------------------------------------------------------------------

int num_func_calls = 0;
int num_hess_calls = 0;
class gmm_registration_model
{

public:
    typedef ::column_vector column_vector;
    typedef matrix<float> general_matrix;

    gmm_registration_model(const gmm_utils::GMM3f& source_gmm,
                           const gmm_utils::GMM3f& target_gmm):
      source_gmm_(source_gmm), target_gmm_(target_gmm)
    {
      std::vector<Eigen::Matrix3f> dRdu;
      std::vector<std::vector<Eigen::Matrix3f>> d2Rdu2;
      Eigen::Matrix<float, 6, 1> grad;
      Eigen::Matrix<float, 6, 6> H;
      Eigen::Vector3f x_opt = Eigen::Vector3f::Zero();
      corr_grad_hess(Eigen::Matrix3f::Identity(), x_opt,
                     target_gmm_.getNClusters(),
                     source_gmm_.getNClusters(),
                     target_gmm_.getWeights(),
                     target_gmm_.getWeights(),
                     target_gmm_.getCovs(),
                     target_gmm_.getCovs(),
                     target_gmm_.getMeans(),
                     target_gmm_.getMeans(),
                     1.0,
                     dRdu, d2Rdu2,
                     false,
                     Et_, grad, H);
    }

    // Evaluate cost function at x
    float operator() (
        const column_vector& x
    ) const {
      Eigen::Vector3f t_ = Eigen::Vector3f(x(0), x(1), x(2));
      Eigen::Vector3f u(x(3), x(4), x(5));
      float unorm = u.norm();
      num_func_calls++;
      Eigen::Matrix3f R_;
      if (unorm < 1e-10)
        R_ = Eigen::Matrix3f::Identity();
      else
        R_ = Eigen::AngleAxisf(unorm, u.normalized()).toRotationMatrix();

      std::vector<Eigen::Matrix3f> dRdu;
      std::vector<std::vector<Eigen::Matrix3f>> d2Rdu2;
      Eigen::Matrix<float, 6, 1> grad;
      Eigen::Matrix<float, 6, 6> H;
      float fval;

      corr_grad_hess(R_, t_,
                     target_gmm_.getNClusters(),
                     source_gmm_.getNClusters(),
                     target_gmm_.getWeights(),
                     source_gmm_.getWeights(),
                     target_gmm_.getCovs(),
                     source_gmm_.getCovs(),
                     target_gmm_.getMeans(),
                     source_gmm_.getMeans(),
                     Et_,
                     dRdu, d2Rdu2,
                     false,
                     fval, grad, H);
      //std::cout << "Got f["<<num_func_calls<<"] = " << fval << std::endl;

      return fval;
    }

    void get_derivative_and_hessian (
        const column_vector& x,
        column_vector& der,
        general_matrix& hess
    ) const
    {
        Eigen::Vector3f t_ = Eigen::Vector3f(x(0), x(1), x(2));
        Eigen::Vector3f u(x(3), x(4), x(5));
        Eigen::Matrix3f R_;
        float unorm = u.norm();
        if (unorm < 1e-10)
          R_ = Eigen::Matrix3f::Identity();
        else
          R_ = Eigen::AngleAxisf(unorm, u.normalized()).toRotationMatrix();
        num_hess_calls++;
        Eigen::Matrix<float, 6, 1> grad;
        Eigen::Matrix<float, 6, 6> H;
        std::vector<Eigen::Matrix3f> dRdu;
        std::vector<std::vector<Eigen::Matrix3f>> d2Rdu2;

        partial_wrt_u(x(3), x(4), x(5), dRdu, d2Rdu2);
        float fval;
        corr_grad_hess(R_, t_,
                       target_gmm_.getNClusters(),
                       source_gmm_.getNClusters(),
                       target_gmm_.getWeights(),
                       source_gmm_.getWeights(),
                       target_gmm_.getCovs(),
                       source_gmm_.getCovs(),
                       target_gmm_.getMeans(),
                       source_gmm_.getMeans(),
                       Et_,
                       dRdu, d2Rdu2,
                       true,
                       fval, grad, H);
        der  = dlib::mat(grad);
        hess = dlib::mat(H);
        //std::cin.ignore();
    }
private:
  const gmm_utils::GMM3f & source_gmm_;
  const gmm_utils::GMM3f & target_gmm_;
  float Et_;

  void corr_grad_hess(const Eigen::Matrix3f& Ri, const Eigen::Vector3f& ti,
                      const uint32_t& Nm,
                      const uint32_t& Nk,
                      const Eigen::MatrixXf& wm,
                      const Eigen::MatrixXf& wk,
                      const Eigen::MatrixXf& Lm,
                      const Eigen::MatrixXf& Omk,
                      const Eigen::MatrixXf& mu_m,
                      const Eigen::MatrixXf& nu_k,
                      const float& Et,
                      const std::vector<Eigen::Matrix3f>& dRdu,
                      const std::vector<std::vector<Eigen::Matrix3f> >& d2Rdu2,
                      const bool& compute_grad_hess,
                      float& score, Eigen::Matrix<float, 6, 1>& J,
                      Eigen::Matrix<float, 6, 6>& H) const
  {
    uint32_t Nmk = Nm * Nk;

    Eigen::Vector3f dFdt;
    Eigen::Matrix3f dFdR;
    Eigen::Matrix<float, 3, 6> Htr;
    Eigen::Matrix<float, 9, 3> Delta, Gamma;

    score = 0;
    dFdt.setZero(); dFdR.setZero();
    Delta.setZero(); Gamma.setZero();
    Htr.setZero();

    // Generate rotation matrix
    for(uint32_t mk = 0; mk < Nmk; mk++)
    {

      // Extract component parameters
      uint32_t k_ind = mk % Nk;
      uint32_t m_ind = mk / Nk;
      auto Om = Eigen::Map<const Eigen::Matrix3f>(Omk.col(k_ind).data(), NUM_DIMS, NUM_DIMS);
      auto L  = Eigen::Map<const Eigen::Matrix3f>(Lm.col(m_ind).data(), NUM_DIMS, NUM_DIMS);
      auto nu = nu_k.col(k_ind);
      auto mu = mu_m.col(m_ind);
      float w = wm(m_ind) * wk(k_ind);

      // Compute ymk
      Eigen::Vector3f Rnu = Ri * nu;
      Eigen::Vector3f ymk = mu - Rnu - ti;
      // Compute R*iOm
      Eigen::Matrix3f ROmi = Ri * Om;


      // Compute Smk
      Eigen::Matrix3f ROmiR = ROmi * Ri.transpose(); // R * Omk * R'
      Eigen::Matrix3f Smki = L + ROmiR;

      // Compute determinant and inverse
      Eigen::Matrix3f Smk;
      float det;
      bool invertible;

      double detd;
      Eigen::Matrix3d Smkd = Smk.cast<double>();
      Eigen::Matrix3d Smkid = Smki.cast<double>();
      Smkid.computeInverseAndDetWithCheck(Smkd, detd, invertible, 1e-30);
      if (!invertible)
      {
        std::cout << "Matrix " << mk << " was not invertible. Det = " << det
                  << ", mat: \n" << Smki << std::endl;
	Smkid = Smkd.inverse();
	std::cout << "Smkid: " << Smkid << std::endl;
	std::cout << "Smk: " << Smk << std::endl;
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Smkid, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Smkd = svd.matrixU() * svd.singularValues().asDiagonal().inverse() * svd.matrixV().transpose();
	std::cerr << "svd.matrixU(): " << svd.matrixU() << std::endl;
	std::cerr << "svd.singularValues(): " << (svd.singularValues()) << std::endl;
	std::cerr << "svd.matrixV(): " << svd.matrixV() << std::endl;
	std::cerr << "SVD Smkd: " << Smkd << std::endl;
        continue;
      }
      det = (float) detd;
      Smki = Smkid.cast<float>();
      Smk = Smkd.cast<float>();

      // Compute Symk
      Eigen::Vector3f Symk = Smk * ymk ;

      // Compute fmk
      const float PI_32 = 1.f/(2*M_PI*sqrt(2*M_PI));

      float fmk = w * PI_32 * (1/sqrt(det)) * exp( -0.5f * ymk.transpose() * Symk );
      fmk = fmk / Et;
      score += fmk;

      // Jacobian and Hessian for the rotation matrix wrt axis angle components
      // Evaluate gradient
      // Compute partial derivative wrt t
      dFdt.noalias() += fmk * Symk;

      // Compute partial derivative wrt R
      Eigen::Matrix3f Synut_mk = Symk * nu.transpose();
      Eigen::Matrix3f Sy_ytSmk = Symk * Symk.transpose();
      Eigen::Matrix3f Sy_ytSROmi = Sy_ytSmk * ROmi;
      Eigen::Matrix3f SmkROmi = Smk * ROmi;

      dFdR.noalias() += fmk * (Synut_mk + Sy_ytSROmi - SmkROmi);

      if (compute_grad_hess)
      {
        // Evaluate Hessian
        // Compute dt^2 terms
        Htr.block<3,3>(0,0).noalias() += fmk * (Sy_ytSmk - Smk);

        for (int  ix = 0; ix < NUM_DIMS; ix++)
        {
          Eigen::Matrix3f A = dRdu[ix];
          Eigen::Matrix3f ROmiAt = ROmi * A.transpose();

          Eigen::Matrix3f Za = ROmiAt + ROmiAt.transpose();

          Eigen::Vector3f Anu = A * nu;

          Eigen::Vector3f ZaSymk = Za * Symk;
          Eigen::Matrix3f SmkROmiAt = Smk * ROmiAt;

          float da  = -SmkROmiAt.trace();

          float qa = (ZaSymk + 2 * Anu).transpose() *  Symk;

          float da_hqa = da + 0.5f * qa;
          Eigen::Vector3f S__ZaSy_Anu_mk = Smk * (ZaSymk + Anu);

          // dtdr
          Eigen::Vector3f dtdr_ix = fmk * (da_hqa * Symk - S__ZaSy_Anu_mk);
          Htr.block<NUM_DIMS, 1>(0,3+ix).noalias() += dtdr_ix;

          // dr^2
          Eigen::Matrix3f OmAt = Om * A.transpose();
          Eigen::Matrix3f OmiRtSZa_mk = SmkROmi.transpose() * Za;
          Eigen::Matrix3f Dba = (-OmAt + OmiRtSZa_mk) * Smk;

          Eigen::Matrix3f da_qa__db_plus_da_qa__qb = 2 * da_hqa *
            (-SmkROmi + Sy_ytSROmi + Synut_mk).transpose();

          // T
          Eigen::Matrix3f nunutAt = nu * Anu.transpose();
          Eigen::Matrix3f Anyt = Anu * ymk.transpose();

          // Terms from partial of q_ra wrt to rb containing only first derivatives
          // of R wrt rb.
          Eigen::Matrix3f ZaSyyt = ZaSymk * ymk.transpose();
          Eigen::Matrix3f dqadrb_b =
            -2 * nu * ZaSymk.transpose() * Smk
            - 2 * SmkROmi.transpose() * (ZaSyyt + ZaSyyt.transpose()) *Smk
            + 2 * OmAt * Sy_ytSmk
            - 2 * SmkROmi.transpose() * (Anyt + Anyt.transpose()) * Smk
            - 2 * nunutAt * Smk;

          Delta.block<3, 3>(3*ix, 0) += 0.5 * fmk * (2*Dba + da_qa__db_plus_da_qa__qb + dqadrb_b);

          Gamma.block<3, 3>(3*ix, 0) += fmk * (-SmkROmi.transpose() + (Sy_ytSROmi + Synut_mk).transpose());
        }
      }
    }
    if (compute_grad_hess)
    {
      J.topRows(NUM_DIMS) = dFdt;
      for (int ix = 0; ix < NUM_DIMS; ix++)
        J(NUM_DIMS+ix) = (dFdR.transpose() * dRdu[ix]).trace();

      H.topLeftCorner(NUM_DIMS, NUM_DIMS) = Htr.topLeftCorner<3,3>();
      H.topRightCorner(NUM_DIMS, NUM_DIMS) = Htr.topRightCorner<3,3>();
      H.bottomLeftCorner(NUM_DIMS, NUM_DIMS) = Htr.topRightCorner<3,3>().transpose();

      for (int  ix = 0; ix < NUM_DIMS; ix++)
        for (int  kx = 0; kx < NUM_DIMS; kx++)
          H(NUM_DIMS+ix, NUM_DIMS+kx) = (Delta.block<3,3>(3*kx, 0) * dRdu[ix]).trace() +
            (Gamma.block<3,3>(3*kx, 0) * d2Rdu2[ix][kx]).trace();
    }
  }

  inline void partial_wrt_u(const float& u1, const float& u2,
                            const float& u3, std::vector<Eigen::Matrix3f>& J) const
  {
    std::vector<std::vector<Eigen::Matrix3f>> H_tmp;
    partial_wrt_u(u1, u2, u3, false, J, H_tmp);
  }

  inline void partial_wrt_u(const float& u1, const float& u2,
                            const float& u3, std::vector<Eigen::Matrix3f>& J, std::vector<std::vector<Eigen::Matrix3f>>& H) const
  {
    partial_wrt_u(u1, u2, u3, true, J, H);
  }
  inline void partial_wrt_u(const float& u1, const float& u2,
                            const float& u3,
                            const bool& compute_hessian_flag,
                            std::vector<Eigen::Matrix3f>& J, std::vector<std::vector<Eigen::Matrix3f> >& H) const
  {
    float t3 = u1*u1;
    float t4 = u2*u2;
    float t5 = u3*u3;
    float t6 = t3+t4+t5;
    float t7 = sqrt(t6);
    // Return simplified gradient and Hessian
    if (t7 < 1e-10)
    {
      float J_arr[] =
        {0.f,  0.f,  0.f,    0.f, 0.f,  1.f,   0.f,-1.f, 0.f,
         0.f,  0.f, -1.f,    0.f, 0.f,  0.f,   1.f, 0.f, 0.f,
         0.f,  1.f,  0.f,   -1.f, 0.f,  0.f,   0.f, 0.f, 0.f};
      J.resize(NUM_DIMS);
      for(int ix = 0; ix < NUM_DIMS; ix++)
        J[ix] = Eigen::Map<Eigen::Matrix3f>(J_arr + ix*NUM_DIMS*NUM_DIMS);
      if (compute_hessian_flag)
      {
        float H_arr[] =
          {0.f,  0.f,  0.f,  0.f,  -1.f,  0.f,   0.f,  0.f,  -1.f,
           0.f, 0.5f,  0.f, 0.5f,   0.f,  0.f,   0.f,  0.f,   0.f,
           0.f,  0.f, 0.5f,  0.f,   0.f,  0.f,  0.5f,  0.f,   0.f,
           0.f, 0.5f,  0.f, 0.5f,   0.f,  0.f,   0.f,  0.f,   0.f,
           -1.f,  0.f,  0.f,  0.f,  0.0f,  0.f,   0.f,  0.f,  -1.f,
           0.f,  0.f,  0.f,  0.f,   0.f, 0.5f,   0.f, 0.5f,   0.f,
           0.f,  0.f, 0.5f,  0.f,   0.f,  0.f,  0.5f,  0.f,   0.f,
           0.f,  0.f,  0.f,  0.f,   0.f, 0.5f,   0.f, 0.5f,   0.f,
           -1.f,  0.f,  0.f,  0.f,  -1.f,  0.f,   0.f,  0.f,   0.f};
        H.resize(NUM_DIMS);
        for(int ix = 0; ix < NUM_DIMS; ix++)
        {
          H[ix].resize(NUM_DIMS);
          for(int jx = 0; jx < NUM_DIMS; jx++)
          {
            uint32_t offset = NUM_DIMS * NUM_DIMS *(ix + jx *  NUM_DIMS);
            H[ix][jx] = Eigen::Map<Eigen::Matrix3f>(H_arr + offset);
          }
        }
      }
      return;
    }
    float t8 = t7*(1.0f/2.0f);
    float t2 = sin(t8);
    float t9 = t2*t2;
    float tt6 = t6*t6;
    float t10 = 1.0f/(tt6);
    float t11 = cos(t8);
    float st6 = sqrt(t6);
    float t12 = 1.0f/(t6*st6);
    float t13 = 1.0f/t6;
    float t14 = 1.0f/st6;
    float t15 = t9*t13*u1*2.0f;
    float t16 = t11*t11;
    float t17 = t2*t11*t12*u1*u2*u3*2.0f;
    float t18 = t9*t13*u2*u3;
    float t19 = t2*t11*t12*u2*u3*2.0f;
    float t20 = t9*t13*u2*2.0f;
    float t21 = t9*t13*u1*u3;
    float t22 = t2*t11*t12*u1*u3*2.0f;
    float t23 = t2*t3*t11*t12*u2*2.0f;
    float t24 = t13*t16*u2*u3;
    float t25 = t2*t4*t11*t12*u1*2.0f;
    float t26 = t2*t11*t14*2.0f;
    float t27 = t5*t9*t13;
    float t28 = t2*t5*t11*t12*2.0f;
    float t29 = t4*t9*t10*u1*2.0f;
    float t30 = t5*t9*t10*u1*2.0f;
    float t31 = t2*t3*t11*t12*u1;
    float t32 = t4*t9*t10*u2*2.0f;
    float t33 = t5*t9*t10*u2*2.0f;
    float t34 = t2*t3*t11*t12*u2;
    float t35 = t9*t13*u3*2.0f;
    float t36 = t5*t9*t10*u3*2.0f;
    float t37 = t4*t9*t10*u3*2.0f;
    float t38 = t2*t3*t11*t12*u3;
    float t39 = t13*t16*u1*u2;
    float t40 = t13*t16*u1*u3;
    float t41 = t9*t13*u1*u2;
    float t42 = t2*t11*t12*u1*u2*2.0f;
    float t43 = t2*t3*t11*t12*u3*2.0f;
    float t44 = t4*t13*t16;
    float t45 = t2*t5*t11*t12*u1*2.0f;
    float t46 = t3*t9*t13;
    float t47 = t2*t3*t11*t12*2.0f;
    float t48 = t2*t4*t11*t12*u3*2.0f;
    float t49 = t2*t5*t11*t12*u2*2.0f;
    float t50 = t3*t9*t10*u1*2.0f;
    float t51 = t2*t4*t11*t12*u1;
    float t52 = t3*t9*t10*u2*2.0f;
    float t53 = t2*t4*t11*t12*u2;
    float t54 = t3*t9*t10*u3*2.0f;
    float t55 = t2*t4*t11*t12*u3;
    float t56 = t3*t3;
    float t57 = 1.0f/(tt6*t6);
    float t58 = 1.0f/(tt6*st6);//.^(5.0f/2.0f);
    float t59 = t3*t10*t16*u1*u2*(1.0f/2.0f);
    float t60 = t4*t9*t10*u1*u2*(1.0f/2.0f);
    float t61 = t3*t9*t57*u1*u2*8.0f;
    float t62 = t9*t13*u1*u2*(1.0f/2.0f);
    float t63 = t2*t11*t12*u1*u2;
    float t64 = t5*t9*t10*u1*u2*(1.0f/2.0f);
    float t65 = t2*t4*t11*t58*u1*u2*5.0f;
    float t66 = t2*t5*t11*t58*u1*u2*5.0f;
    float t172 = t4*t10*t16*u1*u2*(1.0f/2.0f);
    float t173 = t3*t9*t10*u1*u2*(1.0f/2.0f);
    float t174 = t4*t9*t57*u1*u2*8.0f;
    float t175 = t13*t16*u1*u2*(1.0f/2.0f);
    float t176 = t5*t10*t16*u1*u2*(1.0f/2.0f);
    float t177 = t5*t9*t57*u1*u2*8.0f;
    float t178 = t2*t3*t11*t58*u1*u2*5.0f;
    float t67 = t59+t60+t61+t62+t63+t64+t65+t66-t172-t173-t174-t175-t176-t177-t178;
    float t68 = t9*t13*2.0f;
    float t69 = t5*t9*t10*2.0f;
    float t70 = t4*t4;
    float t71 = t3*t4*t9*t10*(1.0f/2.0f);
    float t72 = t2*t3*t4*t11*t58*5.0f;
    float t73 = t3*t10*t16*u1*u3*(1.0f/2.0f);
    float t74 = t5*t9*t10*u1*u3*(1.0f/2.0f);
    float t75 = t3*t9*t57*u1*u3*8.0f;
    float t76 = t9*t13*u1*u3*(1.0f/2.0f);
    float t77 = t2*t11*t12*u1*u3;
    float t78 = t4*t9*t10*u1*u3*(1.0f/2.0f);
    float t79 = t2*t5*t11*t58*u1*u3*5.0f;
    float t80 = t2*t4*t11*t58*u1*u3*5.0f;
    float t179 = t5*t10*t16*u1*u3*(1.0f/2.0f);
    float t180 = t3*t9*t10*u1*u3*(1.0f/2.0f);
    float t181 = t5*t9*t57*u1*u3*8.0f;
    float t182 = t13*t16*u1*u3*(1.0f/2.0f);
    float t183 = t4*t10*t16*u1*u3*(1.0f/2.0f);
    float t184 = t4*t9*t57*u1*u3*8.0f;
    float t185 = t2*t3*t11*t58*u1*u3*5.0f;
    float t81 = t73+t74+t75+t76+t77+t78+t79+t80-t179-t180-t181-t182-t183-t184-t185;
    float t82 = t5*t9*t10*u2*u3*(1.0f/2.0f);
    float t83 = t4*t9*t10*u2*u3*(1.0f/2.0f);
    float t84 = t9*t13*u2*u3*(1.0f/2.0f);
    float t85 = t9*t10*u2*u3*8.0f;
    float t86 = t3*t10*t16*u2*u3*(1.0f/2.0f);
    float t87 = t3*t9*t57*u2*u3*8.0f;
    float t88 = t2*t5*t11*t58*u2*u3*5.0f;
    float t89 = t2*t4*t11*t58*u2*u3*5.0f;
    float t191 = t5*t10*t16*u2*u3*(1.0f/2.0f);
    float t192 = t4*t10*t16*u2*u3*(1.0f/2.0f);
    float t193 = t5*t9*t57*u2*u3*8.0f;
    float t194 = t4*t9*t57*u2*u3*8.0f;
    float t195 = t13*t16*u2*u3*(1.0f/2.0f);
    float t196 = t3*t9*t10*u2*u3*(1.0f/2.0f);
    float t197 = t2*t3*t11*t58*u2*u3*5.0f;
    float t90 = t82+t83+t84+t85+t86+t87+t88+t89-t191-t192-t193-t194-t195-t196-t197-t2*t11*t12*u2*u3*3.0f;
    float t91 = t4*t9*t10*2.0f;
    float t92 = t5*t5;
    float t93 = t2*t3*t11*t12;
    float t94 = t3*t5*t9*t10*(1.0f/2.0f);
    float t95 = t4*t5*t9*t10*(1.0f/2.0f);
    float t96 = t2*t3*t5*t11*t58*5.0f;
    float t97 = t2*t4*t5*t11*t58*5.0f;
    float t98 = t2*t4*t11*t12*2.0f;
    float J_arr[] =
      {t15+t29+t30+t31-t3*t9*t10*u1*2.0f-t2*t11*t14*u1-t2*t4*t11*t12*u1-t2*t5*t11*t12*u1,
       t20-t21-t22+t23+t40-t3*t9*t10*u2*4.0f,
       t35-t39+t41+t42+t43-t3*t9*t10*u3*4.0f,
       t20+t21+t22+t23-t3*t9*t10*u2*4.0f-t13*t16*u1*u3,
       -t15-t29+t30-t31+t50+t51-t2*t11*t14*u1-t2*t5*t11*t12*u1,
       t17+t26-t46-t47+t3*t13*t16-t9*t10*u1*u2*u3*4.0f,
       t35+t39+t43-t3*t9*t10*u3*4.0f-t9*t13*u1*u2-t2*t11*t12*u1*u2*2.0f,
       t17-t26+t46+t47-t3*t13*t16-t9*t10*u1*u2*u3*4.0f,
       -t15+t29-t30-t31+t50-t51-t2*t11*t14*u1+t2*t5*t11*t12*u1,

       t32+t33+t34-t9*t13*u2*2.0f-t3*t9*t10*u2*2.0f-t2*t11*t14*u2-t2*t4*t11*t12*u2-t2*t5*t11*t12*u2,
       t15-t18-t19+t24+t25-t4*t9*t10*u1*4.0f,
       t17-t26-t44+t98+t4*t9*t13-t9*t10*u1*u2*u3*4.0f,
       t15+t18+t19+t25-t4*t9*t10*u1*4.0f-t13*t16*u2*u3,
       t20-t32+t33-t34+t52+t53-t2*t11*t14*u2-t2*t5*t11*t12*u2,
       t35+t39-t41-t42+t48-t4*t9*t10*u3*4.0f,
       t17+t26+t44-t4*t9*t13-t2*t4*t11*t12*2.0f-t9*t10*u1*u2*u3*4.0f,
       t35-t39+t41+t42+t48-t4*t9*t10*u3*4.0f,
       -t20+t32-t33-t34+t52-t53-t2*t11*t14*u2+t2*t5*t11*t12*u2,

       t36+t37+t38-t9*t13*u3*2.0f-t3*t9*t10*u3*2.0f-t2*t11*t14*u3-t2*t4*t11*t12*u3-t2*t5*t11*t12*u3,
       t17+t26-t27-t28+t5*t13*t16-t9*t10*u1*u2*u3*4.0f,
       t15+t18+t19-t24+t45-t5*t9*t10*u1*4.0f,
       t17+t27+t28-t2*t11*t14*2.0f-t5*t13*t16-t9*t10*u1*u2*u3*4.0f,
       -t35+t36-t37-t38+t54+t55-t2*t11*t14*u3-t2*t5*t11*t12*u3,
       t20-t21-t22+t40+t49-t5*t9*t10*u2*4.0f,
       t15-t18-t19+t24+t45-t5*t9*t10*u1*4.0f,
       t20+t21+t22-t40+t49-t5*t9*t10*u2*4.0f,
       t35-t36+t37-t38+t54-t55-t2*t11*t14*u3+t2*t5*t11*t12*u3};
    J.resize(NUM_DIMS);
    for(int ix = 0; ix < NUM_DIMS; ix++)
      J[ix] = Eigen::Map<Eigen::Matrix3f>(J_arr + ix*NUM_DIMS*NUM_DIMS);
    if (compute_hessian_flag)
    {
      float t99 = t3*t4*t10*t16;
      float t100 = t3*t4*t9*t57*16.f;
      float t101 = t10*t16*u1*u2*u3*3.0f;
      float t123 = t3*t9*t10*4.0f;
      float t124 = t9*t10*u1*u2*u3*3.0f;
      float t125 = t2*t11*t58*u1*u2*u3*6.0f;
      float t148 = t4*t9*t10*4.0f;
      float t149 = t3*t4*t9*t10;
      float t150 = t2*t3*t4*t11*t58*10.f;
      float t102 = t17+t47+t68+t98+t99+t100+t101-t123-t124-t125-t148-t149-t150;
      float t103 = t9*t13*u3;
      float t104 = t2*t11*t12*u3*2.0f;
      float t105 = t2*t11*t12*u1*u2*6.0f;
      float t106 = t9*t13*u1;
      float t107 = t2*t11*t12*u1*2.0f;
      float t108 = t5*t10*t16*u1*3.0f;
      float t109 = t3*t10*t16*u2*u3;
      float t110 = t3*t9*t57*u2*u3*16.f;
      float t119 = t13*t16*u1;
      float t120 = t9*t10*u2*u3*4.0f;
      float t121 = t3*t9*t10*u2*u3;
      float t122 = t2*t3*t11*t58*u2*u3*10.f;
      float t151 = t5*t9*t10*u1*3.0f;
      float t152 = t2*t5*t11*t58*u1*6.0f;
      float t111 = t19+t45+t106+t107+t108+t109+t110-t119-t120-t121-t122-t151-t152;
      float t112 = t9*t13*u2;
      float t113 = t2*t11*t12*u2*2.0f;
      float t114 = t5*t10*t16*u2*3.0f;
      float t115 = t4*t10*t16*u1*u3;
      float t116 = t4*t9*t57*u1*u3*16.f;
      float t118 = t13*t16*u2;
      float t129 = t9*t10*u1*u3*4.0f;
      float t130 = t4*t9*t10*u1*u3;
      float t131 = t2*t4*t11*t58*u1*u3*10.f;
      float t142 = t5*t9*t10*u2*3.0f;
      float t144 = t2*t5*t11*t58*u2*6.0f;
      float t117 = t22+t49+t112+t113+t114+t115+t116-t118-t129-t130-t131-t142-t144;
      float t126 = t4*t9*t10*u1*3.0f;
      float t127 = t2*t4*t11*t58*u1*6.0f;
      float t214 = t4*t10*t16*u1*3.0f;
      float t128 = t19-t25-t106-t107+t109+t110+t119-t120-t121-t122+t126+t127-t214;
      float t132 = t4*t10*t16*u3*3.0f;
      float t133 = t5*t10*t16*u1*u2;
      float t134 = t5*t9*t57*u1*u2*16.f;
      float t135 = t3*t5*t10*t16;
      float t136 = t3*t5*t9*t57*16.f;
      float t215 = t5*t9*t10*4.0f;
      float t223 = t3*t5*t9*t10;
      float t224 = t2*t3*t5*t11*t58*10.f;
      float t137 = -t17+t28+t47+t68-t101-t123+t124+t125+t135+t136-t215-t223-t224;
      float t138 = t13*t16*u3;
      float t139 = t4*t9*t10*u3*3.0f;
      float t140 = t2*t4*t11*t58*u3*6.0f;
      float t162 = t9*t10*u1*u2*4.0f;
      float t163 = t5*t9*t10*u1*u2;
      float t164 = t2*t5*t11*t58*u1*u2*10.f;
      float t141 = t42-t48-t103-t104-t132+t133+t134+t138+t139+t140-t162-t163-t164;
      float t143 = t2*t11*t12*u1*u3*6.0f;
      float t145 = t3*t10*t16*u3*3.0f;
      float t146 = t3*t10*t16*u1*u2;
      float t147 = t3*t9*t57*u1*u2*16.f;
      float t153 = -t17+t47+t68+t98+t99+t100-t101-t123+t124+t125-t148-t149-t150;
      float t154 = t4*t10*t16*u1*u2;
      float t155 = t4*t9*t57*u1*u2*16.f;
      float t156 = t19-t45-t106-t107-t108+t109+t110+t119-t120-t121-t122+t151+t152;
      float t157 = t22-t49-t112-t113-t114+t115+t116+t118-t129-t130-t131+t142+t144;
      float t158 = t9*t13*u3*3.0f;
      float t159 = t5*t10*t16*u3*3.0f;
      float t160 = t2*t11*t12*u3*6.0f;
      float t161 = t2*t5*t11*t12*u3*2.0f;
      float t165 = t10*t16*t56*(1.0f/2.0f);
      float t166 = t3*t9*t13*(1.0f/2.0f);
      float t167 = t9*t56*t57*8.0f;
      float t168 = t3*t4*t10*t16*(1.0f/2.0f);
      float t169 = t3*t5*t10*t16*(1.0f/2.0f);
      float t170 = t3*t4*t9*t57*8.0f;
      float t171 = t3*t5*t9*t57*8.0f;
      float t186 = -t59-t60-t61+t62+t63+t64-t65+t66+t172+t173+t174-t175-t176-t177+t178;
      float t187 = t4*t9*t13*(1.0f/2.0f);
      float t188 = t4*t9*t10*10.f;
      float t189 = t9*t10*t70*(1.0f/2.0f);
      float t190 = t2*t11*t58*t70*5.0f;
      float t198 = t9*t10*u1*u3*8.0f;
      float t199 = -t73+t74-t75+t76-t78+t79-t80-t179+t180-t181-t182+t183+t184+t185+t198-t2*t11*t12*u1*u3*3.0f;
      float t200 = t2*t11*t12*u2*u3;
      float t201 = t82-t83+t84-t86-t87+t88-t89-t191+t192-t193+t194-t195+t196+t197+t200;
      float t202 = t3*t9*t10*2.0f;
      float t203 = t5*t9*t13*(1.0f/2.0f);
      float t204 = t5*t9*t10*10.f;
      float t205 = t9*t10*t92*(1.0f/2.0f);
      float t206 = t2*t4*t11*t12;
      float t207 = t2*t11*t58*t92*5.0f;
      float t208 = t3*t9*t10*u2*3.0f;
      float t209 = t2*t3*t11*t58*u2*6.0f;
      float t210 = t3*t9*t10*u3*3.0f;
      float t211 = t2*t3*t11*t58*u3*6.0f;
      float t212 = t3*t10*t16*u2*3.0f;
      float t213 = t22+t23+t112+t113+t115+t116-t118-t129-t130-t131-t208-t209+t212;
      float t216 = t42+t43+t103+t104+t133+t134-t138+t145-t162-t163-t164-t210-t211;
      float t217 = t4*t5*t10*t16;
      float t218 = t4*t5*t9*t57*16.f;
      float t240 = t4*t5*t9*t10;
      float t241 = t2*t4*t5*t11*t58*10.f;
      float t219 = t17+t28+t68+t98+t101-t124-t125-t148-t215+t217+t218-t240-t241;
      float t220 = t2*t11*t12*u2*u3*6.0f;
      float t221 = t3*t10*t16*u1*u3;
      float t222 = t3*t9*t57*u1*u3*16.f;
      float t225 = t19+t25+t106+t107+t109+t110-t119-t120-t121-t122-t126-t127+t214;
      float t226 = t13*t16*u2*3.0f;
      float t227 = t4*t9*t10*u2*3.0f;
      float t228 = t2*t4*t11*t58*u2*6.0f;
      float t229 = t17+t28+t47+t68+t101-t123-t124-t125+t135+t136-t215-t223-t224;
      float t230 = t42+t48+t103+t104+t132+t133+t134-t138-t139-t140-t162-t163-t164;
      float t231 = t5*t10*t16*u1*u3;
      float t232 = t5*t9*t57*u1*u3*16.f;
      float t233 = t9*t13*u1*3.0f;
      float t234 = t3*t10*t16*u1*3.0f;
      float t235 = t2*t11*t12*u1*6.0f;
      float t236 = t2*t3*t11*t12*u1*2.0f;
      float t237 = t22-t23-t112-t113+t115+t116+t118-t129-t130-t131+t208+t209-t212;
      float t238 = t4*t10*t16*u2*u3;
      float t239 = t4*t9*t57*u2*u3*16.f;
      float t242 = t42-t43-t103-t104+t133+t134+t138-t145-t162-t163-t164+t210+t211;
      float t243 = -t17+t28+t68+t98-t101+t124+t125-t148-t215+t217+t218-t240-t241;
      float t244 = t5*t10*t16*u2*u3;
      float t245 = t5*t9*t57*u2*u3*16.f;
      float t246 = t3*t9*t10*10.f;
      float t247 = t9*t10*t56*(1.0f/2.0f);
      float t248 = t2*t11*t56*t58*5.0f;
      float t249 = t9*t10*u1*u2*8.0f;
      float t250 = -t59+t60-t61+t62-t64+t65-t66-t172+t173-t174-t175+t176+t177+t178+t249-t2*t11*t12*u1*u2*3.0f;
      float t251 = t10*t16*t70*(1.0f/2.0f);
      float t252 = t9*t57*t70*8.0f;
      float t253 = t2*t5*t11*t12;
      float t254 = t4*t5*t10*t16*(1.0f/2.0f);
      float t255 = t4*t5*t9*t57*8.0f;
      float t256 = -t73-t74-t75+t76+t77+t78-t79+t80+t179+t180+t181-t182-t183-t184+t185;
      float t257 = -t82+t83+t84-t86-t87-t88+t89+t191-t192+t193-t194-t195+t196+t197+t200;
      float H_arr[] =
        {t68+t69+t71+t72+t91+t94+t96+t165+t166+t167-t3*t9*t10*10.f-t2*t11*t14 -
         t3*t13*t16*(1.0f/2.0f)-t9*t10*t56*(1.0f/2.0f)+t2*t3*t11*t12*6.0f -
         t2*t4*t11*t12-t2*t5*t11*t12-t3*t4*t10*t16*(1.0f/2.0f)-t3*t5*t10*t16*(1.0f/2.0f) -
         t3*t4*t9*t57*8.0f-t3*t5*t9*t57*8.0f-t2*t11*t56*t58*5.0f,
         -t43-t103-t104+t105+t138-t145+t146+t147+t210+t211-t9*t10*u1*u2*12.f-t3*t9*t10*u1*u2-t2*t3*t11*t58*u1*u2*10.f,
         t23+t112+t113-t118+t143-t208-t209+t212+t221+t222-t9*t10*u1*u3*12.f-t3*t9*t10*u1*u3-t2*t3*t11*t58*u1*u3*10.f,

         t43+t103+t104+t105+t145+t146+t147-t13*t16*u3-t3*t9*t10*u3*3.0f-t9*t10*u1*u2*12.f -
         t2*t3*t11*t58*u3*6.0f-t3*t9*t10*u1*u2-t2*t3*t11*t58*u1*u2*10.f,
         -t68+t69-t71-t72-t91+t94+t96-t165+t166-t167+t168-t169+t170-t171+t206+t246+t247+t248 -
         t2*t11*t14-t3*t13*t16*(1.0f/2.0f)-t2*t3*t11*t12*4.0f-t2*t5*t11*t12,
         t19+t109+t110-t120-t121-t122-t233-t234-t235-t236+t13*t16*u1*3.0f+t3*t9*t10*u1*3.0f+t2*t3*t11*t58*u1*6.0f,

         -t23-t112-t113+t118+t143+t208+t209+t221+t222-t3*t10*t16*u2*3.0f-t9*t10*u1*u3*12.f -
         t3*t9*t10*u1*u3-t2*t3*t11*t58*u1*u3*10.f,
         t19+t109+t110-t120-t121-t122+t233+t234+t235+t236-t13*t16*u1*3.0f-t3*t9*t10*u1*3.0f-t2*t3*t11*t58*u1*6.0f,
         -t68-t69+t71+t72+t91-t94-t96-t165+t166-t167-t168+t169-t170+t171-t206+t246+t247+t248+t253 -
         t2*t11*t14-t3*t13*t16*(1.0f/2.0f)-t2*t3*t11*t12*4.0f,


         t67, t153, t225,
         t102, t186, t237,
         t128, t213, t250,


         t81,  t156, t229,
         t111, t199, t242,
         t137, t216, t256,


         t67,  t153, t225,
         t102, t186, t237,
         t128, t213, t250,


         -t68+t69-t71-t72+t93+t95+t97+t168+t170+t187+t188+t189+t190-t3*t9*t10*2.0f -
         t2*t11*t14-t4*t13*t16*(1.0f/2.0f)-t10*t16*t70*(1.0f/2.0f)-t9*t57*t70*8.0f -
         t2*t4*t11*t12*4.0f-t2*t5*t11*t12-t4*t5*t10*t16*(1.0f/2.0f)-t4*t5*t9*t57*8.0f,
         -t48-t103-t104+t105-t132+t138+t139+t140+t154+t155-t9*t10*u1*u2*12.f-t4*t9*t10*u1*u2-t2*t4*t11*t58*u1*u2*10.f,
         t22+t115+t116-t129-t130-t131-t226-t227-t228+t9*t13*u2*3.0f+t2*t11*t12*u2*6.0f+t4*t10*t16*u2*3.0f+t2*t4*t11*t12*u2*2.0f,

         t48+t103+t104+t105+t132+t154+t155-t13*t16*u3-t4*t9*t10*u3*3.0f-t9*t10*u1*u2*12.f -
         t2*t4*t11*t58*u3*6.0f-t4*t9*t10*u1*u2-t2*t4*t11*t58*u1*u2*10.f,
         t68+t69+t71+t72-t93+t95+t97-t168-t170+t187-t188-t189-t190+t202+t251+t252-t2*t11*t14 -
         t4*t13*t16*(1.0f/2.0f)+t2*t4*t11*t12*6.0f-t2*t5*t11*t12-t4*t5*t10*t16*(1.0f/2.0f)-t4*t5*t9*t57*8.0f,
         -t25-t106-t107+t119+t126+t127-t214+t220+t238+t239-t9*t10*u2*u3*12.f-t4*t9*t10*u2*u3-t2*t4*t11*t58*u2*u3*10.f,

         t22+t115+t116-t129-t130-t131+t226+t227+t228-t9*t13*u2*3.0f-t2*t11*t12*u2*6.0f-t4*t10*t16*u2*3.0f-t2*t4*t11*t12*u2*2.0f,
         t25+t106+t107-t119-t126-t127+t214+t220+t238+t239-t9*t10*u2*u3*12.f-t4*t9*t10*u2*u3-t2*t4*t11*t58*u2*u3*10.f,
         -t68-t69+t71+t72-t93-t95-t97-t168-t170+t187+t188+t189+t190+t202-t251-t252+t253+t254+t255 -
         t2*t11*t14-t4*t13*t16*(1.0f/2.0f)-t2*t4*t11*t12*4.0f,


         t90,  t157, t230,
         t117, t201, t243,
         t141, t219, t257,


         t81,  t156, t229,
         t111, t199, t242,
         t137, t216, t256,


         t90,  t157, t230,
         t117, t201, t243,
         t141, t219, t257,


         -t68+t91+t93-t94+t95-t96+t97+t169+t171+t203+t204+t205+t207-t3*t9*t10*2.0f -
         t2*t11*t14-t5*t13*t16*(1.0f/2.0f)-t10*t16*t92*(1.0f/2.0f)-t9*t57*t92*8.0f -
         t2*t4*t11*t12-t2*t5*t11*t12*4.0f-t4*t5*t10*t16*(1.0f/2.0f)-t4*t5*t9*t57*8.0f,
         t42+t133+t134-t158-t159-t160-t161-t162-t163-t164+t13*t16*u3*3.0f+t5*t9*t10*u3*3.0f+t2*t5*t11*t58*u3*6.0f,
         t49+t112+t113+t114-t118-t142+t143-t144+t231+t232-t9*t10*u1*u3*12.f-t5*t9*t10*u1*u3-t2*t5*t11*t58*u1*u3*10.f,

         t42+t133+t134+t158+t159+t160+t161-t13*t16*u3*3.0f-t5*t9*t10*u3*3.0f-t9*t10*u1*u2*4.0f -
         t2*t5*t11*t58*u3*6.0f-t5*t9*t10*u1*u2-t2*t5*t11*t58*u1*u2*10.f,
         -t68-t91-t93+t94-t95+t96-t97-t169-t171+t202+t203+t204+t205+t206+t207+t254+t255 -
         t2*t11*t14-t5*t13*t16*(1.0f/2.0f)-t10*t16*t92*(1.0f/2.0f)-t9*t57*t92*8.0f-t2*t5*t11*t12*4.0f,
         -t45-t106-t107-t108+t119+t151+t152+t220+t244+t245-t9*t10*u2*u3*12.f-t5*t9*t10*u2*u3-t2*t5*t11*t58*u2*u3*10.f,

         -t49-t112-t113-t114+t118+t142+t143+t144+t231+t232-t9*t10*u1*u3*12.f-t5*t9*t10*u1*u3-t2*t5*t11*t58*u1*u3*10.f,
         t45+t106+t107+t108-t119-t151-t152+t220+t244+t245-t9*t10*u2*u3*12.f-t5*t9*t10*u2*u3-t2*t5*t11*t58*u2*u3*10.f,
         t68+t91-t93+t94+t95+t96+t97-t169-t171+t202+t203-t204-t205-t206-t207-t254-t255-t2*t11*t14-
         t5*t13*t16*(1.0f/2.0f)+t10*t16*t92*(1.0f/2.0f)+t9*t57*t92*8.0f+t2*t5*t11*t12*6.0f};

      H.resize(NUM_DIMS);
      for(int ix = 0; ix < NUM_DIMS; ix++)
        H[ix].resize(NUM_DIMS);
      for(int ix = 0; ix < NUM_DIMS; ix++)
      {
        for(int jx = 0; jx < NUM_DIMS; jx++)
        {
          uint32_t offset = NUM_DIMS * NUM_DIMS *(jx + ix *  NUM_DIMS);
          H[jx][ix] = Eigen::Map<Eigen::Matrix3f>(H_arr + offset);
        }
      }
    }
  }
};

class MatcherD2D
{
public:
  float match(const gmm_utils::GMM3f& source_gmm, const gmm_utils::GMM3f& target_gmm,
             const Eigen::Transform<float,3,Eigen::Affine,Eigen::ColMajor>& Tinit,
             Eigen::Transform<float,3,Eigen::Affine,Eigen::ColMajor>& Tout)
  {
    // Convert transformation matrix to axis angle and translation vector representation
    column_vector x(6);
    Eigen::Vector3f t = Tinit.translation();
    Eigen::AngleAxisf aa(Tinit.rotation());
    Eigen::Vector3f u = aa.axis() * aa.angle();
    x = t(0), t(1), t(2), u(0), u(1), u(2);

    Eigen::Matrix<float, 1,6> tmp;
    tmp << t(0), t(1), t(2), u(0), u(1), u(2);
    //std::cout << "Initializing with " << tmp << std::endl;
    float min = find_max_trust_region(objective_delta_stop_strategy(1e-7),
                                gmm_registration_model(source_gmm, target_gmm),
                                x,   //
                                5.0); // Initial trust region radius

    // Convert axis angle representation to rotation matrix
    u = Eigen::Vector3f(x(3), x(4), x(5));
    Tout =  Eigen::Translation<float,3>( x(0), x(1), x(2))*
            Eigen::AngleAxis<float>(u.norm(),u.normalized());
    //std::cerr << "Tout: " << Tout.matrix() << std::endl;
    //std::cout << "Got " << num_func_calls << " function calls and "
    //                    << num_hess_calls << " Hessian calls" << std::endl;
    return min;
  }
};
