#include <boost /math/distributions/students_t.hpp>

void calc_t (unsigned long ∗H[ 2 ], double * range, double * t_ret ,double * t_dof_ret ,double * t_p_ret ) {
	double mean[ 2 ] = { 0.0 , 0.0 };
	double var[ 2 ] = { 0.0 , 0.0 };
	double n[ 2 ] = { 0.0 , 0.0 };

	// only calculate for bins which are non−zero−> faster for real measurements
	vector<int> nonZeroBins;
	for(int idx_bin = 0; idx_bin < number_of_bins; idx_bin++) {
		bool isNonZero = false;
		for( size_t idx_category = 0; idx_category < number_of_categories; idx_category++) {
		 if(H[ idx_category ] [ idx_bin ] != 0)
		  isNonZero = true;
		 }

		if( isNonZero )
		 nonZeroBins.push_back ( idx_bin );
		}

	for( size_t idx_category = 0; idx_category < number_of_categories; idx_category++) {
		for each (auto idx_bin in nonZeroBins ){
		 mean[ idx_category ] += H[ idx_category ][ idx_bin ] ∗ range [ idx_bin ];
		 n[ idx_category ] += H[ idx_category ] [ idx_bin ];
		}

		mean[ idx_category ] = mean[ idx_category ] / n[ idx_category ];

		for each(auto idx_bin in nonZeroBins ){
		 double temp = ( range [ idx_bin ]−mean[ idx_category ] );
		 var[ idx_category ] += (temp * temp) ∗ H[ idx_category ][ idx_bin ];
		}

		var[ idx_category ] = var[ idx_category ] / n[ idx_category ];
	}


	// calculate t−value
	double mean_diff = mean[ 0 ]−mean[ 1 ];
	double variance_sum = ( var[ 0 ] / n[ 0 ] ) + ( var[ 1 ] / n[ 1 ] );
	∗t_ret = mean_diff / sqrt ( variance_sum );

	// calculate degree of freedom
	double denominator = (( var[ 0 ] / n[ 0 ] ) ∗ ( var[ 0 ] / n[ 0 ] ) ) / (n[ 0 ]−1) + ((var[ 1 ] / n[ 1 ] )∗( var[ 1 ] / n[ 1 ] ) ) / (n[ 1 ]−1);
    ∗t_dof_ret = ( variance_sum * variance_sum ) / denominator;
    boost :: math :: students_t_distribution <> t_dist (∗t_dof_ret );
    ∗t_p_ret = 2∗boost :: math :: cdf ( t_dist ,−fabs (∗t_ret ) );
 }