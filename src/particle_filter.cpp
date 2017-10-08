/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

double gaussian_2d(double x1, double x2, double mu1, double mu2, double sig1, double sig2);

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	default_random_engine gen;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++)
	{

		double w = 1.0; // initial weight
		weights.push_back(w);

		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		while (p.theta> M_PI) 
			p.theta-= 2*M_PI;
		while (p.theta< -M_PI) 
			p.theta+= 2*M_PI;
		p.weight = w;
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	if (abs(yaw_rate) > 1e-4)
	{
		for (int i = 0; i < num_particles; i++)
		{
			double dtheta = yaw_rate * delta_t;
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + dtheta) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + dtheta)) + dist_y(gen);
			particles[i].theta += dtheta + dist_theta(gen);
		}
	}
	else
	{
		for (int i = 0; i < num_particles; i++)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++)
	{
		double d2_min = 1E100;
		double d2 = 0;
		for (int j = 0; j < predicted.size(); j++)
		{
			d2 = (observations[i].x - predicted[j].x) * (observations[i].x - predicted[j].x) + (observations[i].y - predicted[j].y) * (observations[i].y - predicted[j].y);
			if (d2 < d2_min)
			{
				d2_min = d2;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	std::vector<LandmarkObs> obs_world;
	std::vector<LandmarkObs> landmarks;
	std::vector<int> assc;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
	weights.clear();
	// turn obs to world corrd

	for (int j = 0; j < num_particles; j++)
	{

		Particle p = particles[j];
		sense_x.clear();
		sense_y.clear();
		obs_world.clear(); //observations transformed to world coordinate
		landmarks.clear(); //landmarks seen by this particle
		for (int i = 0; i < observations.size(); i++)
		{
			LandmarkObs ob;
			ob.id = observations[i].id;
			ob.x = observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta) + p.x;
			ob.y = observations[i].x * sin(p.theta) + observations[i].y * cos(p.theta) + p.y;
			obs_world.push_back(ob);
			sense_x.push_back(ob.x);
			sense_y.push_back(ob.y);
		}

		for (int i = 0; i < map_landmarks.landmark_list.size(); i++)
		{
			double d2 = (map_landmarks.landmark_list[i].x_f - p.x) * (map_landmarks.landmark_list[i].x_f - p.x) + (map_landmarks.landmark_list[i].y_f - p.y) * (map_landmarks.landmark_list[i].y_f - p.y);

			if (sqrt(d2) < sensor_range)
			{
				LandmarkObs ob;
				ob.id = map_landmarks.landmark_list[i].id_i;
				ob.x = map_landmarks.landmark_list[i].x_f;
				ob.y = map_landmarks.landmark_list[i].y_f;
				landmarks.push_back(ob);
			}
		}
		dataAssociation(landmarks, obs_world);
		assc.clear();
		for (int i = 0; i < obs_world.size(); i++)
		{
			assc.push_back(obs_world[i].id);
		}
		particles[j] = SetAssociations(particles[j], assc, sense_x, sense_y);

		particles[j].weight = 1;
		for (int i = 0; i < obs_world.size(); i++)
		{
			double x_f_ = map_landmarks.landmark_list[obs_world[i].id-1].x_f;
			double y_f_ = map_landmarks.landmark_list[obs_world[i].id-1].y_f;
			particles[j].weight *= gaussian_2d(obs_world[i].x, obs_world[i].y,
											   x_f_, y_f_, sig_x, sig_y);
		}
		weights.push_back(particles[j].weight);
	}
	// normalize weights
	double alpha = 0;
	for (int i=0; i<weights.size(); i++)
		alpha +=weights[i];
	for (int i=0; i<num_particles;i++)
	{
		weights[i] = weights[i]/alpha;
		particles[i].weight = particles[i].weight/alpha;
	}

}

double gaussian_2d(double x1, double x2, double mu1, double mu2, double sig1, double sig2)
{

	double gauss_norm = 1.0 / (2 * M_PI * sig1 * sig2);

	double exponent = (x1 - mu1) * (x1 - mu1) / (2 * sig1 * sig1) + (x2 - mu2) * (x2 - mu2) / (2 * sig2 * sig2);
	return gauss_norm * exp(-exponent);
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	int N = num_particles;
	int ind = rand()%N;
	double beta =0;
	double wmax =*max_element(weights.begin(), weights.end());
	std::uniform_real_distribution<double> unif(0,wmax*2);
	std::default_random_engine gen;
	// double a_random_double = unif(gen);
	std::vector<Particle> new_particles;
	for (int i=0; i<N; i++)
	{
		beta += unif(gen);
		while (weights[ind]<beta)
		{
			beta-= weights[ind];
			ind = (ind+1)%N;
		}
		new_particles.push_back(particles[ind]);
	}
	particles=new_particles;
	weights.clear();
	for (int i=0; i<particles.size();i++)
	{
		weights.push_back(particles[i].weight);
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
