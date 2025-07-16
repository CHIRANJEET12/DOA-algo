#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>
#include <ctime>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

random_device rd;
mt19937 gen(rd());

// ELD Problem Constants
const int N = 3;
const double Pd = 850;
const double P_min[N] = {100, 50, 100};
const double P_max[N] = {600, 200, 400};
const double e[N] = {300, 150, 200};
const double f[N] = {0.0315, 0.063, 0.042};
const double a[N] = {0.001562, 0.00482, 0.00194};
const double b[N] = {7.92, 7.97, 7.85};
const double c[N] = {561, 78, 310};

// Enhanced objective function
double ELD_cost(const vector<double>& P, int iter = 0, int max_iter = 1) {
    double cost = 0.0;
    double totalPower = 0.0;
    
    // Main cost calculation
    for (int i = 0; i < N; ++i) {
        double valve = e[i] * sin(f[i] * (P_min[i] - P[i]));
        cost += a[i]*P[i]*P[i] + b[i]*P[i] + c[i] + fabs(valve);
        totalPower += P[i];
    }
    
    // Dynamic penalty that increases with iterations
    double penalty = (1000 + 5000*(double)iter/max_iter) * fabs(totalPower - Pd);
    return cost + penalty;
}

// Initialization function
vector<vector<double>> initialization(int agents, int dim, const vector<double>& ub, const vector<double>& lb) {
    vector<vector<double>> pop(agents, vector<double>(dim));
    for (int i = 0; i < dim; ++i) {
        uniform_real_distribution<> dist(lb[i], ub[i]);
        for (int j = 0; j < agents; ++j) {
            pop[j][i] = dist(gen);
        }
    }
    return pop;
}

pair<double, vector<double>> DOA(int pop, int T, const vector<double>& lb, const vector<double>& ub, vector<double>& convergence) {
    const int D = N;
    vector<vector<double>> x = initialization(pop, D, ub, lb);
    vector<double> sbest(D);
    double fbest = numeric_limits<double>::infinity();
    
    convergence.clear();
    convergence.reserve(T/10);

    double w_min = 0.4, w_max = 0.9;
    
    for (int i = 0; i < T; ++i) {
        double w = w_max - (w_max-w_min)*i/T;
        
        for (int j = 0; j < pop; ++j) {
            double fit = ELD_cost(x[j], i, T);
            
            if (fit < fbest) {
                fbest = fit;
                sbest = x[j];
            }
            
            uniform_real_distribution<> dist(0,1);
            int k = max(1, (int)ceil(D*(0.3 + 0.5*dist(gen))));
            
            vector<int> indices(D);
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);
            
            for (int h = 0; h < k; ++h) {
                int idx = indices[h];
                double step = w * (cos((i+1)*M_PI/T) + 1) * dist(gen);
                x[j][idx] += step * (ub[idx]-lb[idx]);
                
                if (x[j][idx] > ub[idx]) x[j][idx] = 2*ub[idx] - x[j][idx];
                if (x[j][idx] < lb[idx]) x[j][idx] = 2*lb[idx] - x[j][idx];
                x[j][idx] = max(lb[idx], min(ub[idx], x[j][idx]));
            }
        }
        
        if (i % 10 == 0) {
            convergence.push_back(fbest);
        }

        if (i % 50 == 0) {
            vector<double> candidate = sbest;
            uniform_real_distribution<> local_dist(-0.01, 0.01);
            for (int j = 0; j < D; ++j) {
                candidate[j] += local_dist(gen) * (ub[j]-lb[j]);
                candidate[j] = max(lb[j], min(ub[j], candidate[j]));
            }
            double candidate_fit = ELD_cost(candidate, i, T);
            if (candidate_fit < fbest) {
                fbest = candidate_fit;
                sbest = candidate;
            }
        }
    }
    
    double totalPower = accumulate(sbest.begin(), sbest.end(), 0.0);
    if (fabs(totalPower - Pd) > 1e-3) {
        double scale = Pd / totalPower;
        for (double &p : sbest) p *= scale;
        fbest = ELD_cost(sbest, T, T);
    }
    
    return {fbest, sbest};
}

// Function to save convergence data to a file
void saveConvergenceData(const vector<double>& convergence, const string& filename) {
    ofstream outfile(filename);
    if (!outfile) {
        cerr << "Error opening file for convergence data!" << endl;
        return;
    }
    
    outfile << "Iteration,BestCost\n";
    for (size_t i = 0; i < convergence.size(); ++i) {
        outfile << (i*10) << "," << fixed << setprecision(6) << convergence[i] << "\n";
    }
    outfile.close();
    cout << "Convergence data saved to " << filename << endl;
}

int main() {
    int pop = 150;
    int max_iter = 3000;
    
    vector<double> lb(P_min, P_min + N);
    vector<double> ub(P_max, P_max + N);
    
    vector<double> convergence_data;
    
    clock_t start = clock();
    auto result = DOA(pop, max_iter, lb, ub, convergence_data);
    clock_t end = clock();
    
    saveConvergenceData(convergence_data, "doa_convergence.csv");
    
    cout << "\nOptimized ELD Solution:\n";
    double totalPower = 0;
    for (int i = 0; i < N; ++i) {
        cout << "P" << i+1 << " = " << fixed << setprecision(2) 
             << result.second[i] << " MW\n";
        totalPower += result.second[i];
    }
    
    cout << "Total Power: " << totalPower << " MW (Demand: " << Pd << " MW)\n";
    cout << "Minimum Cost: $" << fixed << setprecision(2) << result.first << "/hr\n";
    cout << "Computation Time: " << double(end-start)/CLOCKS_PER_SEC << " sec\n";
    
    return 0;
}