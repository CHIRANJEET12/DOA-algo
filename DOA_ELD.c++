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

// ELD Constants
const int N = 13;
const double Pd = 1800;
const double P_min[N] = {0, 0, 0, 60, 60, 60, 60, 60, 60, 40, 40, 55, 55};
const double P_max[N] = {680, 360, 360, 180, 180, 180, 180, 180, 180, 120, 120, 120, 120};
const double A[N] = {0.000284, 0.000056, 0.000056, 0.000324, 0.000324, 0.000324, 0.000324, 0.000324, 0.000324, 0.000324, 0.000324, 0.000284, 0.000284};
const double b[N] = {8.10, 8.10, 8.10, 7.74, 7.74, 7.74, 7.74, 7.74, 7.74, 8.60, 8.60, 8.60, 8.60};
const double c[N] = {550, 309, 307, 240, 240, 240, 240, 240, 240, 126, 126, 126, 126};
const double E[N] = {300, 200, 150, 150, 150, 150, 150, 150, 150, 100, 100, 100, 100};
const double f[N] = {0.035, 0.042, 0.042, 0.063, 0.063, 0.063, 0.063, 0.063, 0.063, 0.084, 0.084, 0.084, 0.084};

double ELD_cost(const vector<double>& P) {
    double cost = 0.0;
    double totalPower = 0.0;
    for (int i = 0; i < N; ++i) {
        double valve = E[i] * sin(f[i] * (P_min[i] - P[i]));
        cost += A[i] * P[i] * P[i] + b[i] * P[i] + c[i] + fabs(valve);
        totalPower += P[i];
    }
    // Penalty for power balance violation
    double penalty = 1e5 * fabs(totalPower - Pd);
    return cost + penalty;
}

vector<vector<double>> initialization(int agents, int dim, const vector<double>& ub, const vector<double>& lb) {
    vector<vector<double>> pop(agents, vector<double>(dim));
    for (int i = 0; i < dim; ++i) {
        uniform_real_distribution<> dist(lb[i], ub[i]);
        for (int j = 0; j < agents; ++j)
            pop[j][i] = dist(gen);
    }
    return pop;
}

// Constrain to balance total power
void balancePower(vector<double>& P) {
    double sumP = accumulate(P.begin(), P.end(), 0.0);
    double scale = Pd / sumP;
    for (int i = 0; i < N; ++i) {
        P[i] *= scale;
        P[i] = min(P_max[i], max(P_min[i], P[i]));
    }
}

pair<double, vector<double>> DOA(int pop, int T, const vector<double>& lb, const vector<double>& ub, vector<double>& convergence) {
    const int D = N;
    vector<vector<double>> x = initialization(pop, D, ub, lb);

    // Balance initial population
    for (auto& agent : x)
        balancePower(agent);

    vector<double> sbest(D);
    double fbest = numeric_limits<double>::infinity();
    convergence.clear();
    convergence.reserve(T / 10);

    uniform_real_distribution<> rand01(0, 1);

    double w_min = 0.4, w_max = 0.9;

    for (int i = 0; i < T; ++i) {
        double w = w_max * pow((1.0 - (double)i / T), 1.5);

        for (int j = 0; j < pop; ++j) {
            double fit = ELD_cost(x[j]);
            if (fit < fbest) {
                fbest = fit;
                sbest = x[j];
            }

            int k = max(1, (int)ceil(D * (0.3 + 0.5 * rand01(gen))));
            vector<int> indices(D);
            iota(indices.begin(), indices.end(), 0);
            shuffle(indices.begin(), indices.end(), gen);

            for (int h = 0; h < k; ++h) {
                int idx = indices[h];
                double step = w * cos((i + 1) * M_PI / T) * rand01(gen) * (ub[idx] - lb[idx]) * 0.25;
                x[j][idx] += step;

                if (x[j][idx] > ub[idx]) x[j][idx] = ub[idx] - rand01(gen) * (ub[idx] - lb[idx]) * 0.1;
                if (x[j][idx] < lb[idx]) x[j][idx] = lb[idx] + rand01(gen) * (ub[idx] - lb[idx]) * 0.1;
            }

            balancePower(x[j]);
        }

        if (i % 10 == 0)
            convergence.push_back(fbest);

        // Local search refinement
        if (i % 20 == 0) {
            vector<double> candidate = sbest;
            uniform_real_distribution<> local(-0.01, 0.01);
            for (int j = 0; j < D; ++j) {
                candidate[j] += local(gen) * (ub[j] - lb[j]);
                candidate[j] = max(lb[j], min(ub[j], candidate[j]));
            }
            balancePower(candidate);

            double cand_fit = ELD_cost(candidate);
            if (cand_fit < fbest) {
                fbest = cand_fit;
                sbest = candidate;
            }
        }

        // Elitism every 20 iterations
        if (i % 20 == 0) {
            int worst_idx = 0;
            double worst_fit = -1e9;
            for (int j = 0; j < pop; ++j) {
                double fit = ELD_cost(x[j]);
                if (fit > worst_fit) {
                    worst_fit = fit;
                    worst_idx = j;
                }
            }
            x[worst_idx] = sbest;
        }
    }

    balancePower(sbest);
    fbest = ELD_cost(sbest);

    return {fbest, sbest};
}

void saveConvergenceData(const vector<double>& convergence, const string& filename) {
    ofstream outfile(filename);
    if (!outfile) {
        cerr << "Error opening file for convergence data!" << endl;
        return;
    }

    outfile << "Iteration,BestCost\n";
    for (size_t i = 0; i < convergence.size(); ++i)
        outfile << (i * 10) << "," << fixed << setprecision(6) << convergence[i] << "\n";

    outfile.close();
    cout << "Convergence data saved to " << filename << endl;
}

double calculateSD(const vector<double>& data, double mean) {
    double sumSqDiff = 0.0;
    for (double x : data) {
        sumSqDiff += pow(x - mean, 2);
    }
    return sqrt(sumSqDiff / data.size());
}

// Helper to calculate median
double calculateMedian(vector<double> data) {
    sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0)
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    else
        return data[n / 2];
}

int main()
{
    int pop = 50;
    int max_iter = 500;
    int runs = 50;

    vector<double> lb(P_min, P_min + N);
    vector<double> ub(P_max, P_max + N);

    vector<double> best_convergence;

    vector<double> all_costs;
    vector<vector<double>> all_solutions;
    double best_cost = 1e9;
    vector<double> best_solution;

    clock_t total_time = 0;

    for (int r = 0; r < runs; ++r) {
        vector<double> current_convergence;

        clock_t start = clock();
        auto result = DOA(pop, max_iter, lb, ub, current_convergence);
        clock_t end = clock();
        total_time += (end - start);

        all_costs.push_back(result.first);
        all_solutions.push_back(result.second);

        if (result.first < best_cost) {
            best_cost = result.first;
            best_solution = result.second;
            best_convergence = current_convergence;
        }
    }

    double mean_cost = accumulate(all_costs.begin(), all_costs.end(), 0.0) / runs;
    double median_cost = calculateMedian(all_costs);
    double std_dev = calculateSD(all_costs, mean_cost);

    // Save convergence of best run
    saveConvergenceData(best_convergence, "doa_convergence_best.csv");

    cout << "\nBest Optimized ELD Solution:\n";
    double totalPower = 0;
    for (int i = 0; i < N; ++i) {
        cout << "P" << i + 1 << " = " << fixed << setprecision(2)
             << best_solution[i] << " MW\n";
        totalPower += best_solution[i];
    }

    cout << "Total Power: " << totalPower << " MW (Demand: " << Pd << " MW)\n";
    cout << "Best Cost: $" << fixed << setprecision(2) << best_cost << "/hr\n";
    cout << "Mean Cost: $" << fixed << setprecision(2) << mean_cost << "/hr\n";
    cout << "Median Cost: $" << fixed << setprecision(2) << median_cost << "/hr\n";
    cout << "Std Deviation: $" << fixed << setprecision(2) << std_dev << "/hr\n";
    cout << "Average Computation Time: " << double(total_time) / runs / CLOCKS_PER_SEC << " sec per run\n";

    return 0;
}
