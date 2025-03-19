#include <math.h>
#include "sad.h"

// Function to compute the Sum of Absolute Differences (SAD) between a time series and a query
void sad(float* ts, float* query, int ts_length, int query_length, float* result) {
    // Calculate the number of possible positions the query can be aligned with the time series
    int length = ts_length - query_length + 1;

    // Iterate over each possible alignment position
    for (int i = 0; i < length; i++) {
        float distance = 0;
        // Compute the SAD for the current alignment position
        for (int j = 0; j < query_length; j++) {
            distance += fabs(ts[i + j] - query[j]);
        }
        // Store the computed distance in the result array
        *result = distance;
    }
}