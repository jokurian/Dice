#include "input.h"
#include "RHF.h"

using namespace std;
using namespace Eigen;

using matPair = std::array<MatrixXcd, 2>;

RHF::RHF(Hamiltonian& ham, bool pleftQ, std::string fname) 
{
  MatrixXd hf = MatrixXd::Zero(ham.norbs, ham.norbs);
  readMat(hf, fname);
  det = hf.block(0, 0, ham.norbs, ham.nalpha);
  detT = det.adjoint();
  leftQ = pleftQ;
  if (leftQ) ham.rotateCholesky(detT, rotChol, rotCholMat, true);
};


void RHF::getSample(std::array<Eigen::MatrixXcd, 2>& sampleDet) 
{
  sampleDet[0] = det;
  sampleDet[1] = det;
};


std::complex<double> RHF::overlap(std::array<Eigen::MatrixXcd, 2>& psi)
{
  complex<double> overlap = (detT * psi[0]).determinant() * (detT * psi[1]).determinant();
  return overlap;
};


std::complex<double> RHF::overlap(Eigen::MatrixXcd& psi)
{
  complex<double> overlap = (detT * psi).determinant();
  overlap *= overlap;
  return overlap;
};


void RHF::forceBias(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  assert(ham.intType == "r"); 
  matPair thetaT;
  thetaT[0] = (psi[0] * (detT * psi[0]).inverse()).transpose();
  thetaT[1] = (psi[1] * (detT * psi[1]).inverse()).transpose();
  MatrixXcd thetaTSA = thetaT[0] + thetaT[1];
  Eigen::Map<VectorXcd> thetaTFlat(thetaTSA.data(), thetaTSA.rows() * thetaTSA.cols());
  fb = thetaTFlat.transpose() * rotCholMat[0];
};


void RHF::forceBias(Eigen::MatrixXcd& psi, Hamiltonian& ham, Eigen::VectorXcd& fb)
{
  assert(ham.intType == "r"); 
  MatrixXcd thetaT;
  thetaT = (psi * (detT * psi).inverse()).transpose();
  Eigen::Map<VectorXcd> thetaTFlat(thetaT.data(), thetaT.rows() * thetaT.cols());
  fb = 2. * thetaTFlat.transpose() * rotCholMat[0];
};


void RHF::oneRDM(std::array<Eigen::MatrixXcd, 2>& psi, Eigen::MatrixXcd& rdmSample) 
{
  rdmSample = (psi[0] * (detT * psi[0]).inverse() * detT).transpose() + (psi[1] * (detT * psi[1]).inverse() * detT).transpose();
};


std::array<std::complex<double>, 2> RHF::hamAndOverlap(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham) 
{ 
  assert(ham.intType == "r"); 
  complex<double> overlap = (detT * psi[0]).determinant() * (detT * psi[1]).determinant();
  complex<double> ene = ham.ecore;
  
  // calculate theta and green
  matPair theta, green;
  theta[0] = psi[0] * (detT * psi[0]).inverse();
  theta[1] = psi[1] * (detT * psi[1]).inverse();
  green[0] = theta[0] * detT;
  green[1] = theta[1] * detT;

  // one body part
  ene += green[0].cwiseProduct(ham.h1).sum() + green[1].cwiseProduct(ham.h1).sum();

  // two body part
  MatrixXcd fup = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  MatrixXcd fdn = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  for (int i = 0; i < ham.ncholEne; i++) {
    fup.noalias() = rotChol[i] * theta[0];
    fdn.noalias() = rotChol[i] * theta[1];
    complex<double> cup = fup.trace();
    complex<double> cdn = fdn.trace();
    ene += (cup * cup + cdn * cdn + 2. * cup * cdn - fup.cwiseProduct(fup.transpose()).sum() - fdn.cwiseProduct(fdn.transpose()).sum()) / 2.;
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};


std::array<std::complex<double>, 2> RHF::hamAndOverlap(Eigen::MatrixXcd& psi, Hamiltonian& ham) 
{ 
  assert(ham.intType == "r"); 
  complex<double> overlap = (detT * psi).determinant();
  overlap *= overlap;
  complex<double> ene = ham.ecore;
  
  // calculate theta and green
  MatrixXcd theta, green;
  theta = psi * (detT * psi).inverse();
  green = theta * detT;

  // one body part
  ene += 2. * green.cwiseProduct(ham.h1).sum();

  // two body part
  MatrixXcd f = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  for (int i = 0; i < ham.ncholEne; i++) {
    f.noalias() = rotChol[i] * theta;
    complex<double> c = f.trace();
    ene += (2. * c * c - f.cwiseProduct(f.transpose()).sum());
  }

  std::array<complex<double>, 2> hamOverlap;
  hamOverlap[0] = ene * overlap;
  hamOverlap[1] = overlap;
  return hamOverlap;
};

std::array<std::complex<double>, 3> RHF::orbitalEnergy(std::array<Eigen::MatrixXcd, 2>& psi, Hamiltonian& ham,int orbital)
{
	return std::array<std::complex<double>, 3>();
};

std::complex<double> RHF::allorbitalEnergy(Eigen::MatrixXcd& psi, Hamiltonian& ham)
{
  MatrixXcd theta, green;
  theta = psi * (detT * psi).inverse();
  green = theta * detT;
  complex<double> ene(0,0); // = ham.ecore; 

  MatrixXcd fup = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  complex<double> e21(0,0);
  complex<double> e22(0,0);
//  complex<double> exx(0,0);
//  exx += green.cwiseProduct(ham.h1).sum()*2.0 ; //+ green[1].cwiseProduct(ham.h1).sum(); 
  for (int i = 0; i<int(ham.nelec/2); i++){
    ene += (green.col(i).cwiseProduct(ham.h1.col(i)).sum())*2.0;
  }
//  cout <<exx<<":"<<ene<<'\n';
  for (int i = 0; i<ham.ncholEne; i++){
    fup.noalias() = rotChol[i] * theta;
    for (int j =0; j<int(ham.nelec/2); j++){
      e21 += (fup(j,j) * 2.0) * (fup.trace() * 2.);
      e22 -= (fup.row(j).cwiseProduct(fup.col(j).transpose()).sum()) * 2.;
    }
  }

  return (ham.ecore+ene + (e21+e22)*0.5);
};


std::array<std::complex<double>,3> RHF::orbitalEnergy(Eigen::MatrixXcd& psi, Hamiltonian& ham,int orbital)
{
  complex<double> overlap = (detT * psi).determinant();
  overlap *= overlap;
  complex<double> e;
  if(orbital==-1){
    e  = allorbitalEnergy(psi,ham);
    std::array<complex<double>, 3> hamOverlapOrbital; 
    hamOverlapOrbital[1] = overlap;
    hamOverlapOrbital[2] = e;
	
	MatrixXcd fup = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
    complex<double> enef  = ham.ecore;
	MatrixXcd theta, green;
	theta = psi * (detT * psi).inverse();
    green = theta * detT;
    
	enef += 2.0 * green.cwiseProduct(ham.h1).sum();
	for (int i = 0; i<ham.ncholEne; i++){
		fup.noalias() = rotChol[i]*theta;
		complex<double> c = fup.trace();
		enef += (2. * c * c - fup.cwiseProduct(fup.transpose()).sum());
	}
	hamOverlapOrbital[0] = enef*overlap;
    return hamOverlapOrbital;
  }

  assert(ham.intType == "r");
  complex<double> enef  = ham.ecore;
  complex<double> eneo(0,0);
  MatrixXcd theta, green;
  theta = psi * (detT * psi).inverse();
  green = theta * detT;


  enef += 2.0 * green.cwiseProduct(ham.h1).sum();
//  eneo += 2.0 * (green.transpose().col(orbital).cwiseProduct(ham.h1.col(orbital)).sum());
  int nocc = ham.nelec/2;
  int norbs = rotChol[0].cols();
  
  // 
  MatrixXcd fup = MatrixXcd::Zero(rotChol[0].rows(), rotChol[0].rows());
  MatrixXcd fup1 = MatrixXcd::Zero(nocc,nocc);
  for (int i = 0; i<ham.ncholEne; i++){
    fup1.noalias() =  rotChol[i].block(0,nocc,nocc,norbs-nocc) * theta.block(nocc,0,norbs-nocc,nocc); //rotChol[i].block(0,nocc-1,nocc,ham.norbs-nocc) * theta.block(nocc-1,0,ham.norbs-nocc,nocc);
    fup.noalias() = rotChol[i]*theta;
    complex<double> c1 = fup1.trace();
    complex<double> c = fup.trace();
    eneo += 1.0*(fup1(orbital,orbital)) * (c1 * 2.); //divided by 2
    eneo -= 1.0*(fup1.row(orbital).cwiseProduct(fup1.col(orbital).transpose()).sum()); //divided by 2
    enef += (2. * c * c - fup.cwiseProduct(fup.transpose()).sum());
  }
  std::array<complex<double>, 3> hamOverlapOrbital;
 
  hamOverlapOrbital[0] = enef * overlap;
  hamOverlapOrbital[1] = overlap;
  hamOverlapOrbital[2] = eneo;
  return hamOverlapOrbital;

};
