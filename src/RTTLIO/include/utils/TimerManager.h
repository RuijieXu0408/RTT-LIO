#include <map>
#include <string>
#include <chrono>
#include <iostream>

class TimerManager
{
private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::map<std::string, double> accumulated_times;
    std::map<std::string, int> call_counts;

public:
    void start(const std::string &name)
    {
        start_times[name] = std::chrono::high_resolution_clock::now();
    }

    void end(const std::string &name)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_times[name])
                            .count() /
                        1000.0; // ms

        accumulated_times[name] += duration;
        call_counts[name]++;

        std::cout << "[TIMING] " << name << ": " << duration << " ms" << std::endl;
    }

    void printSummary()
    {
        std::cout << "\n============== TIMING SUMMARY ==============" << std::endl;
        for (const auto &item : accumulated_times)
        {
            std::string name = item.first;
            double total_time = item.second;
            int count = call_counts[name];
            double avg_time = total_time / count;

            std::cout << name << ":" << std::endl;
            std::cout << "  Total: " << total_time << " ms" << std::endl;
            std::cout << "  Calls: " << count << std::endl;
            std::cout << "  Average: " << avg_time << " ms" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }
};
