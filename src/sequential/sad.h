#ifndef SAD_H
#define SAD_H

// Function to compute the Sum of Absolute Differences (SAD) between a time series and a query
void sad(float* ts, float* query, int ts_length, int query_length, float* result);

#endif