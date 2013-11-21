#include <iostream>
#include <boost/compute.hpp>

namespace compute = boost::compute;

// the point centroid example calculates and displays the
// centroid of a set of 3D points stored as float4's
void mexfunction()
{
	using compute::float4_;

	// point coordinates
	float points[] = { 1.0f, 2.0f, 3.0f, 0.0f,
		-2.0f, -3.0f, 4.0f, 0.0f,
		1.0f, -2.0f, 2.5f, 0.0f,
		-7.0f, -3.0f, -2.0f, 0.0f,
		3.0f, 4.0f, -5.0f, 0.0f };

	// create vector for five points
	compute::vector<float4_> vector(5);

	// copy point data to the device
	compute::copy(
		reinterpret_cast<float4_ *>(points),
		reinterpret_cast<float4_ *>(points) + 5,
		vector.begin()
		);

	// calculate sum
	float4_ sum = compute::accumulate(vector.begin(),
		vector.end(),
		float4_(0, 0, 0, 0));

	// calculate centroid
	float4_ centroid;
	for(size_t i = 0; i < 3; i++){
		centroid[i] = sum[i] / 5.0f;
	}

	// print centroid
	std::cout << "centroid: " << centroid << std::endl;
}