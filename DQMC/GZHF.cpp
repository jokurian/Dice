#include "DQMC/GZHF.h"
#include "utils/input.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

GZHF::GZHF(Hamiltonian &ham, bool pleftQ, std::string fname) {
  int norbs = ham.norbs;
  int nelec = ham.nelec;
  MatrixXcd hf = MatrixXcd::Zero(norbs, norbs);
  readMat(hf, fname);
  detC = hf.block(0, 0, norbs, nelec);
  detCAd = detC.adjoint();
  leftQ = pleftQ;
  if (leftQ) {
    ham.rotateCholesky(detCAd, rotChol, rotCholMat);
  }
};

void GZHF::getSample(Eigen::MatrixXcd &sampleDet) { sampleDet = detC; };

std::complex<double> GZHF::overlap(std::array<Eigen::MatrixXcd, 2> &psi) {
  return std::complex<double>();
};

std::complex<double> GZHF::overlap(Eigen::MatrixXcd &psi) {
  complex<double> overlap;
  overlap = (detCAd * psi).determinant();
  return overlap;
};

void GZHF::forceBias(Eigen::MatrixXcd &psi, Hamiltonian &ham,
                    Eigen::VectorXcd &fb) {
  int norbs = ham.norbs, nelec = ham.nelec;
  MatrixXcd thetaT;

  thetaT = (psi * (detCAd * psi).inverse()).transpose();
  Eigen::Map<VectorXcd> thetaTFlat(thetaT.data(),
                                   thetaT.rows() * thetaT.cols());
  fb = thetaTFlat.transpose() * rotCholMat[0];
};

void GZHF::oneRDM(Eigen::MatrixXcd &psi, Eigen::MatrixXcd &rdmSample) {
  rdmSample = (psi * (detCAd * psi).inverse() * detCAd).transpose();
}

std::array<std::complex<double>, 2>
GZHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2> &psi, Hamiltonian &ham) {
  return std::array<std::complex<double>, 2>();
};

std::array<std::complex<double>, 2> GZHF::hamAndOverlap(Eigen::MatrixXcd &psi,
                                                       Hamiltonian &ham) {
  complex<double> overlap, ene;
  MatrixXcd overlapMat = detCAd * psi;
  overlap = overlapMat.determinant();
  ene = ham.ecore;

  // calculate theta and green
  MatrixXcd theta = psi * overlapMat.inverse();
  MatrixXcd green = (theta * detCAd).transpose();

  // one body part
  ene += green.cwiseProduct(ham.h1soc).sum();

  // two body part
  int norbs = ham.norbs, nelec = ham.nelec;
  MatrixXcd f = MatrixXcd::Zero(nelec, nelec);
  for (int i = 0; i < ham.ncholEne; i++) {
    f.noalias() = rotChol[i] * theta;
    complex<double> c = f.trace();
    ene += (c * c - f.cwiseProduct(f.transpose()).sum()) / 2.;
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};

std::array<std::complex<double>, 3> GZHF::orbitalEnergy(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham,int orbital)
{
return std::array<std::complex<double>, 3>();
};
std::array<std::complex<double>, 3> GZHF::orbitalEnergy(Eigen::MatrixXcd& psi, Hamiltonian& ham,int orbital)
{   
 return std::array<std::complex<double>, 3> ();
};
