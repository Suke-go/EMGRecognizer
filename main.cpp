// -------------------------------------------------------------
// Echo State Network (ESN) – Minimal C++ implementation demo
// Target: single‑channel EMG gesture classification (4 classes)
// Dependencies: Eigen 3.4 (header‑only)
// Compile:
//   g++ -O3 -std=c++17 main.cpp -I/path/to/eigen -o esn_demo
// -------------------------------------------------------------
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// -------------------------------------------
// Utility: one‑hot vector from label (0..C‑1)
// -------------------------------------------
Vector one_hot(size_t label, size_t C)
{
    Vector v = Vector::Zero(C);
    v(label) = 1.0;
    return v;
}

// ------------------------------------------------
// Echo State Network class (leaky‑integrator nodes)
// ------------------------------------------------
class ESN
{
  public:
    ESN(size_t in, size_t res, size_t out,
        double rho=0.9, double scale_in=0.5,
        double leaking=0.3, double ridge=1e-6, double sparsity=0.1)
        : Nin(in), Nres(res), Nout(out), alpha(leaking), lambda(ridge)
    {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> unif(-1.0, 1.0);

        Win  = Matrix::NullaryExpr(Nres, Nin,  [&]{ return scale_in*unif(rng);} );
        Wfb  = Matrix::Zero(Nres, Nout); // no feedback in this demo
        Wres = Matrix::Zero(Nres, Nres);

        // sparse recurrent weights
        for(size_t i=0;i<Nres;++i)
            for(size_t j=0;j<Nres;++j)
                if(unif(rng) < sparsity) Wres(i,j) = unif(rng);
        // spectral radius scaling
        double sr = std::abs(Eigen::EigenSolver<Matrix>(Wres).eigenvalues().real().array()).maxCoeff();
        if(sr>0) Wres *= (rho/sr);

        X = Vector::Zero(Nres);
    }

    // time‑step update
    void step(const Vector &u)
    {
        X = (1.0-alpha)*X + alpha*( (Win*u) + (Wres*X) ).array().tanh();
    }

    // collect state & optional input bias for training
    Vector concat(const Vector &u) const
    {
        Vector z(Nres + Nin + 1);
        z << X, u, 1.0; // bias term
        return z;
    }

    // online buffer for training
    void push_train_sample(const Vector &u, size_t label)
    {
        states.push_back( concat(u) );
        targets.push_back( one_hot(label, Nout) );
    }

    // ridge‑regression closed form
    void train()
    {
        size_t M = states.size();
        size_t D = states[0].size();
        Matrix S(D, M);
        Matrix T(Nout, M);
        for(size_t i=0;i<M;++i)
        {
            S.col(i) = states[i];
            T.col(i) = targets[i];
        }
        Matrix reg = (S*S.transpose() + lambda*Matrix::Identity(D,D)).inverse();
        Wout = T*S.transpose()*reg; // (Nout×D)
    }

    // predict soft scores
    Vector predict(const Vector &u)
    {
        step(u);
        return Wout * concat(u);
    }

  private:
    size_t Nin, Nres, Nout;
    double alpha, lambda;
    Matrix Win, Wres, Wfb, Wout;
    Vector X;
    std::vector<Vector> states;
    std::vector<Vector> targets;
};

// -------------------------
// Example main()
// -------------------------
int main()
{
    constexpr size_t Nin  = 6;   // features per frame (MAV etc.)
    constexpr size_t Nres = 100; // reservoir size
    constexpr size_t Nout = 4;   // gestures

    ESN esn(Nin, Nres, Nout);

    // --- toy training data (replace with real EMG features) ---
    std::vector<Vector> feature_seq;
    std::vector<size_t> labels;
    // here we just load from CSV: label, f1..f6 per line
    std::ifstream ifs("train.csv");
    if(!ifs) { std::cerr << "train.csv not found\n"; return 1; }
    std::string line;
    while(std::getline(ifs,line))
    {
        std::stringstream ss(line);
        size_t lab; char comma;
        ss>>lab>>comma;
        Vector u(Nin);
        for(size_t i=0;i<Nin;++i){ ss>>u[i]; if(i<Nin-1) ss>>comma; }
        esn.step(u);                // warm‑up dynamics
        esn.push_train_sample(u,lab);
    }
    esn.train();

    // --- inference ---
    std::cout << "Enter 6 feature values separated by space (Ctrl+D to quit)\n";
    Vector u(Nin);
    while(std::cin>>u[0])
    {
        for(size_t i=1;i<Nin;++i) std::cin>>u[i];
        Vector y = esn.predict(u);
        size_t pred = 0; y.maxCoeff(&pred);
        std::cout << "Predicted gesture: " << pred << " (scores=" << y.transpose() << ")\n";
    }
    return 0;
}
