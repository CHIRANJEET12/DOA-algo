#include <iostream>
#include <vector>
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

// Random number generator
random_device rd;
mt19937 gen(rd());

// Function prototypes
vector<vector<double>> initialization(int SearchAgents_no, int dim, const vector<double>& ub, const vector<double>& lb);
vector<double> Get_Functions_cec2017(int F, int dim, vector<double>& lb, vector<double>& ub);
double cec17_func(const vector<double>& x, int func_num);
pair<double, vector<double>> DOA(int pop, int T, const vector<double>& lb, const vector<double>& ub, int D, double (*fobj)(const vector<double>&, int), int func_num);

int main() {
    int pop_size = 50;
    int max_iter = 500;
    int run = 10;
    vector<vector<double>> RESULT;

    vector<int> F = {1, 3, 4, 5, 6, 7, 8, 9, 10};
    int variables_no = 10;

    cout << "Currently calculating the CEC2017 function set with a dimensionality of " << variables_no << endl;

    for (size_t func_num = 0; func_num < F.size(); func_num++) {
        cout << "\nF" << F[func_num] << " Function calculation:" << endl;
        int num = F[func_num];
        
        vector<double> lb, ub;
        Get_Functions_cec2017(num, variables_no, lb, ub);
        
        vector<double> final_main(run);

        for (int nrun = 0; nrun < run; nrun++) {
            cout << "Run " << nrun+1 << "/" << run << "..." << flush;
            auto result = DOA(pop_size, max_iter, lb, ub, variables_no, cec17_func, num);
            final_main[nrun] = result.first;
            cout << " Best: " << scientific << setprecision(6) << result.first << endl;
        }

        double min_val = *min_element(final_main.begin(), final_main.end());
        double mean_val = accumulate(final_main.begin(), final_main.end(), 0.0) / final_main.size();
        
        double sum_sq = 0.0;
        for (double val : final_main) {
            sum_sq += (val - mean_val) * (val - mean_val);
        }
        double std_dev = sqrt(sum_sq / final_main.size());
        
        sort(final_main.begin(), final_main.end());
        double median_val = final_main[final_main.size() / 2];
        if (final_main.size() % 2 == 0) {
            median_val = (final_main[final_main.size() / 2 - 1] + final_main[final_main.size() / 2]) / 2.0;
        }
        
        double max_val = *max_element(final_main.begin(), final_main.end());
        
        cout << fixed << setprecision(6);
        cout << "F" << num << " Results:\n";
        cout << "Optimal: " << scientific << min_val << "\nMean: " << mean_val 
             << "\nStd Dev: " << std_dev << "\nMedian: " << median_val 
             << "\nWorst: " << max_val << endl;
    }

    return 0;
}

// DOA algorithm implementation
pair<double, vector<double>> DOA(int pop, int T, const vector<double>& lb, const vector<double>& ub, int D, double (*fobj)(const vector<double>&, int), int func_num) {
    vector<double> lb_adjusted(D), ub_adjusted(D);
    for (int i = 0; i < D; i++) {
        lb_adjusted[i] = lb[i];
        ub_adjusted[i] = ub[i];
    }

    vector<vector<double>> x = initialization(pop, D, ub_adjusted, lb_adjusted);
    vector<int> SELECT(pop);
    iota(SELECT.begin(), SELECT.end(), 0);

    vector<double> sbest(D, 1.0);
    vector<vector<double>> sbestd(5, vector<double>(D, 1.0));
    double fbest = numeric_limits<double>::infinity();
    vector<double> fbestd(5, numeric_limits<double>::infinity());
    vector<double> fbest_history(T, 1.0);

    // Exploration phase
    for (int i = 0; i < (9 * T / 10); i++) {
        for (int m = 0; m < 5; m++) {
            int group_size = pop / 5;
            int start = m * group_size;
            int end = (m + 1) * group_size;
            if (m == 4) end = pop;

            int k_min = max(1, (int)ceil(D / 8.0 / (m + 1)));
            int k_max = max(k_min + 1, (int)ceil(D / 3.0 / (m + 1)));
            uniform_int_distribution<> k_dist(k_min, k_max);
            int k = k_dist(gen);

            // Find best in group
            for (int j = start; j < end; j++) {
                double current_fobj = fobj(x[j], func_num);
                if (current_fobj < fbestd[m]) {
                    sbestd[m] = x[j];
                    fbestd[m] = current_fobj;
                }
            }

            // Update positions
            for (int j = start; j < end; j++) {
                x[j] = sbestd[m];
                vector<int> in(D);
                iota(in.begin(), in.end(), 0);
                shuffle(in.begin(), in.end(), gen);
                
                uniform_real_distribution<> rand_dist(0.0, 1.0);
                if (rand_dist(gen) < 0.9) {
                    for (int h = 0; h < min(k, D); h++) {
                        double rand_val = rand_dist(gen);
                        x[j][in[h]] += (rand_val * (ub_adjusted[in[h]] - lb_adjusted[in[h]]) + lb_adjusted[in[h]]) * 
                                      (cos((1.0*(i+1) + T/10.0) * M_PI/T) + 1)/2.0;

                        // Boundary check
                        if (x[j][in[h]] > ub_adjusted[in[h]] || x[j][in[h]] < lb_adjusted[in[h]]) {
                            if (D > 15) {
                                vector<int> select = SELECT;
                                select.erase(select.begin() + j);
                                uniform_int_distribution<> sel_dist(0, select.size()-1);
                                x[j][in[h]] = x[select[sel_dist(gen)]][in[h]];
                            } else {
                                uniform_real_distribution<> bound_dist(lb_adjusted[in[h]], ub_adjusted[in[h]]);
                                x[j][in[h]] = bound_dist(gen);
                            }
                        }
                    }
                } else {
                    for (int h = 0; h < min(k, D); h++) {
                        uniform_int_distribution<> pop_dist(0, pop-1);
                        x[j][in[h]] = x[pop_dist(gen)][in[h]];
                    }
                }
            }

            if (fbestd[m] < fbest) {
                fbest = fbestd[m];
                sbest = sbestd[m];
            }
        }
    }

    // Exploitation phase
    for (int i = (9 * T / 10); i < T; i++) {
        for (int p = 0; p < pop; p++) {
            double current_fobj = fobj(x[p], func_num);
            if (current_fobj < fbest) {
                sbest = x[p];
                fbest = current_fobj;
            }
        }

        vector<double> fitness(pop);
        for (int j = 0; j < pop; j++) {
            fitness[j] = fobj(x[j], func_num);

            int km = max(2, (int)ceil(D / 3.0));
            uniform_int_distribution<> k_dist(2, km);
            int k = k_dist(gen);

            x[j] = sbest;

            vector<int> in(D);
            iota(in.begin(), in.end(), 0);
            shuffle(in.begin(), in.end(), gen);

            uniform_real_distribution<> rand_dist(0.0, 1.0);
            for (int h = 0; h < min(k, D); h++) {
                x[j][in[h]] += (rand_dist(gen) * (ub_adjusted[in[h]] - lb_adjusted[in[h]]) + lb_adjusted[in[h]]) * 
                               (cos((i + 1) * M_PI / T) + 1) / 2.0;

                if (x[j][in[h]] > ub_adjusted[in[h]] || x[j][in[h]] < lb_adjusted[in[h]]) {
                    if (D > 15) {
                        vector<int> select = SELECT;
                        select.erase(select.begin() + j);
                        uniform_int_distribution<> sel_dist(0, select.size()-1);
                        x[j][in[h]] = x[select[sel_dist(gen)]][in[h]];
                    } else {
                        uniform_real_distribution<> bound_dist(lb_adjusted[in[h]], ub_adjusted[in[h]]);
                        x[j][in[h]] = bound_dist(gen);
                    }
                }
            }
        }
    }

    return make_pair(fbest, sbest);
}

vector<vector<double>> initialization(int SearchAgents_no, int dim, const vector<double>& ub, const vector<double>& lb) {
    vector<vector<double>> Positions(SearchAgents_no, vector<double>(dim));

    if (ub.size() == 1) {
        uniform_real_distribution<> dist(lb[0], ub[0]);
        for (int i = 0; i < SearchAgents_no; i++) {
            for (int j = 0; j < dim; j++) {
                Positions[i][j] = dist(gen);
            }
        }
    } else {
        for (int i = 0; i < dim; i++) {
            uniform_real_distribution<> dist(lb[i], ub[i]);
            for (int j = 0; j < SearchAgents_no; j++) {
                Positions[j][i] = dist(gen);
            }
        }
    }

    return Positions;
}

vector<double> Get_Functions_cec2017(int F, int dim, vector<double>& lb, vector<double>& ub) {
    lb.resize(dim);
    ub.resize(dim);
    fill(lb.begin(), lb.end(), -100.0);
    fill(ub.begin(), ub.end(), 100.0);
    return vector<double>();
}

double cec17_func(const vector<double>& x, int func_num) {
    // Placeholder implementation - replace with actual CEC2017 functions
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}