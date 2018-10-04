import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Gamma, Poisson, Mixture, Categorical, TransformedDistribution, Normal, PointMass

from sklearn.base import BaseEstimator, TransformerMixin

class ZINBayes(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=10, n_mc_samples=1, zero_inflation=True, scalings=True, batch_correction=False, test_iterations=100, optimizer=None, minibatch_size=None, validation=False, X_test=None):
		self.n_components = n_components
		self.est_X = None
		self.est_L = None
		self.est_Z = None

		self.zero_inflation = zero_inflation
		if zero_inflation:
			print('Considering zero-inflation.')

		self.batch_correction = batch_correction
		if batch_correction:
			print('Performing batch correction.')

		self.scalings = scalings
		if scalings:
			print('Considering cell-specific scalings.')

		self.n_mc_samples = n_mc_samples
		self.test_iterations = test_iterations

		self.optimizer = optimizer
		self.minibatch_size = minibatch_size

		# if validation, use X_test to assess convergence
		self.validation = validation and X_test is not None
		self.X_test = X_test
		self.loss_dict = {'t_loss': [], 'v_loss': []}

		sess = ed.get_session()
		sess.close()
		tf.reset_default_graph()

	def close_session(self):
		return self.sess.close()

	def fit(self, X, batch_idx=None, max_iter=100, max_time=60):
		tf.reset_default_graph()

		# Data size
		N = X.shape[0]
		P = X.shape[1]

		if not self.batch_correction:
			batch_idx = None

		# Number of experimental batches
		if batch_idx is not None:
			self.n_batches = np.unique(batch_idx[:, 0]).size
		else:
			self.n_batches = 0

		# Prior for cell scalings
		log_library_size = np.log(np.sum(X, axis=1))
		self.mean_llib, self.std_llib = np.mean(log_library_size), np.std(log_library_size)

		# Create ZINBayes computation graph
		self.define_model(N, P, self.n_components, batch_idx=batch_idx)
		self.inference = self.define_inference(X)

		# If we want to assess convergence during inference on held-out data
		inference_val = None
		if self.validation and self.X_test is not None:
			self.define_val_model(self.X_test.shape[0], P, self.n_components)
			inference_val = self.define_val_inference(self.X_test)

		# Run inference
		self.loss = self.run_inference(self.inference, inference_val=inference_val, n_iterations=max_iter)

		# Get estimated variational distributions of global latent variables
		self.est_qW0 = TransformedDistribution(
	        distribution=Normal(self.qW0.distribution.loc.eval(), self.qW0.distribution.scale.eval()),
	        bijector=tf.contrib.distributions.bijectors.Exp())
		self.est_qr = TransformedDistribution(
	        distribution=Normal(self.qr.distribution.loc.eval(), self.qr.distribution.scale.eval()),
	        bijector=tf.contrib.distributions.bijectors.Exp())
		if self.zero_inflation:
			self.est_qW1 = Normal(self.qW1.loc.eval(), self.qW1.scale.eval())

	def transform(self):
		self.est_Z = self.sess.run(tf.exp(self.qz.distribution.loc))
		if self.scalings:
			self.est_L = self.sess.run(tf.exp(self.ql.distribution.loc))
		self.est_X = self.posterior_nb_mean()

		return self.est_Z

	def fit_transform(self, X, batch_idx=None, max_iter=100, max_time=60):
		self.fit(X, batch_idx=batch_idx, max_iter=max_iter, max_time=60)
		return self.transform()

	def get_est_X(self):
		return self.est_X

	def get_est_l(self):
		return self.est_L

	def score(self, X, batch_idx=None):
		return self.evaluate_loglikelihood(X, batch_idx=batch_idx)

	# def define_model(self, N, P, K, batch_idx=None):
	# 	self.W0 = Gamma(.1 * tf.ones([K + self.n_batches, P]), .3 * tf.ones([K + self.n_batches, P]))

	# 	self.z = Gamma(16. * tf.ones([N, K]), 4. * tf.ones([N, K]))

	# 	self.a = Gamma(2. * tf.ones([1,P]), 1. * tf.ones([1,P]))
	# 	self.r = Gamma(self.a, self.a)

	# 	self.l = Gamma(self.mean_llib**2 / self.std_llib**2 * tf.ones([N, 1]), self.mean_llib / self.std_llib**2 * tf.ones([N, 1]))

	# 	rho = tf.matmul(self.z, self.W0)

	# 	self.likelihood = Poisson(rate=self.r*rho)

	def define_model(self, N, P, K, batch_idx=None):
		self.W0 = Gamma(.1 * tf.ones([K + self.n_batches, P]), .3 * tf.ones([K + self.n_batches, P]))
		if self.zero_inflation:
			self.W1 = Normal(tf.zeros([K + self.n_batches, P]), tf.ones([K + self.n_batches, P]))

		self.z = Gamma(2. * tf.ones([N, K]), 1. * tf.ones([N, K]))

		self.r = Gamma(2. * tf.ones([P,]), 1. * tf.ones([P,]))

		self.l = TransformedDistribution(
		    distribution=Normal(self.mean_llib * tf.ones([N,1]), np.sqrt(self.std_llib) * tf.ones([N,1])),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		if batch_idx is not None and self.n_batches > 0:
			self.rho = tf.matmul(tf.concat([self.z, tf.cast(tf.one_hot(batch_idx[:, 0], self.n_batches), tf.float32)], axis=1), self.W0)
		else:
			self.rho = tf.matmul(self.z, self.W0)

		if self.scalings:
			self.rho = self.rho / tf.reshape(tf.reduce_sum(self.rho, axis=1), (-1, 1)) # NxP
			self.lam = Gamma(self.r, self.r / (self.rho * self.l))
		else:
			self.lam = Gamma(self.r, self.r / self.rho)

		if self.zero_inflation:
			if batch_idx is not None and self.n_batches > 0:
				self.logit_pi = tf.matmul(tf.concat([self.z, tf.cast(tf.one_hot(batch_idx[:, 0], self.n_batches), tf.float32)], axis=1), self.W1)
			else:
				self.logit_pi = tf.matmul(self.z, self.W1)
			self.pi = tf.minimum(tf.maximum(tf.nn.sigmoid(self.logit_pi), 1e-7), 1.-1e-7)
			
			self.cat = Categorical(probs=tf.stack([self.pi, 1.-self.pi], axis=2))

			self.components = [
			    Poisson(rate=1e-30*tf.ones([N, P])),
			    Poisson(rate=self.lam)
			]

			self.likelihood = Mixture(cat=self.cat, components=self.components)
		else:
			self.likelihood = Poisson(rate=self.lam)

	# def define_inference(self, X):
	# 	# Local latent variables
	# 	# self.qz = lognormal_q(self.z.shape)
	# 	self.qz = TransformedDistribution(
	# 	distribution=Normal(tf.Variable(tf.ones(self.z.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.z.shape)))),
	# 	bijector=tf.contrib.distributions.bijectors.Exp())

	# 	self.ql = TransformedDistribution(
	# 	distribution=Normal(tf.Variable(self.mean_llib * tf.ones(self.l.shape)), tf.nn.softplus(tf.Variable(self.std_llib * tf.ones(self.l.shape)))),
	# 	bijector=tf.contrib.distributions.bijectors.Exp())

	# 	self.qr = TransformedDistribution(
	# 	distribution=Normal(tf.Variable(tf.ones(self.r.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.r.shape)))),
	# 	bijector=tf.contrib.distributions.bijectors.Exp())

	# 	self.qa = TransformedDistribution(
	# 	distribution=Normal(tf.Variable(tf.ones(self.a.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.a.shape)))),
	# 	bijector=tf.contrib.distributions.bijectors.Exp())

	# 	self.qW0 = TransformedDistribution(
	# 	distribution=Normal(tf.Variable(tf.ones(self.W0.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.W0.shape)))),
	# 	bijector=tf.contrib.distributions.bijectors.Exp())

	# 	latent_vars_dict = {self.z: self.qz, self.a: self.qa,  self.r: self.qr,  self.W0: self.qW0}

	# 	inference = ed.ReparameterizationKLqp(latent_vars_dict, data={self.likelihood: tf.cast(X, tf.float32)})

	# 	return inference

	def define_inference(self, X):
		# Local latent variables
		# self.qz = lognormal_q(self.z.shape)
		self.qz = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.z.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.z.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		# self.qlam = lognormal_q(self.lam.shape)
		self.qlam = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.lam.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.lam.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		
		# Global latent variables
		# self.qr = lognormal_q(self.r.shape)
		self.qr = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.r.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.r.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		# self.qW0 = lognormal_q(self.W0.shape)
		self.qW0 = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.W0.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.W0.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		
		latent_vars_dict = {self.z: self.qz, self.lam: self.qlam, self.r: self.qr, self.W0: self.qW0}

		if self.zero_inflation:
			self.qW1 = Normal(tf.Variable(tf.zeros(self.W1.shape)), tf.nn.softplus(tf.Variable(0.1*tf.ones(self.W1.shape))))
			latent_vars_dict[self.W1] = self.qW1
		
		if self.scalings:
			# self.ql = lognormal_q(self.l.shape)
			self.ql = TransformedDistribution(
			    distribution=Normal(tf.Variable(self.mean_llib * tf.ones(self.l.shape)), tf.nn.softplus(tf.Variable(self.std_llib * tf.ones(self.l.shape)))),
			    bijector=tf.contrib.distributions.bijectors.Exp())
			latent_vars_dict[self.l] = self.ql
		
		inference = ed.ReparameterizationKLqp(latent_vars_dict, data={self.likelihood: tf.cast(X, tf.float32)})

		return inference

	def define_val_model(self, N, P, K):
		# Define new graph
		self.z_test = Gamma(2. * tf.ones([N, K]), 1. * tf.ones([N, K]))
		self.l_test = TransformedDistribution(
		    distribution=Normal(self.mean_llib * tf.ones([N,1]), np.sqrt(self.std_llib) * tf.ones([N,1])),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		rho_test = tf.matmul(self.z_test, self.W0)
		rho_test = rho_test / tf.reshape(tf.reduce_sum(rho_test, axis=1), (-1, 1)) # NxP

		self.lam_test = Gamma(self.r, self.r / (rho_test * self.l_test))

		if self.zero_inflation:
			logit_pi_test = tf.matmul(self.z_test, self.W1)

			pi_test = tf.minimum(tf.maximum(tf.nn.sigmoid(logit_pi_test), 1e-7), 1.-1e-7)
			cat_test = Categorical(probs=tf.stack([pi_test, 1.-pi_test], axis=2))

			components_test = [
			    Poisson(rate=1e-30 * tf.ones([N, P])),
			    Poisson(rate=self.lam_test)
			]
			self.likelihood_test = Mixture(cat=cat_test, components=components_test)
		else:
			self.likelihood_test = Poisson(rate=self.lam_test)

	def define_val_inference(self, X):
		self.qz_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.z_test.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.z_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		# self.qlam = lognormal_q(self.lam.shape)
		self.qlam_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.lam_test.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.lam_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		# self.ql = lognormal_q(self.l.shape)
		self.ql_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(self.mean_llib * tf.ones(self.l_test.shape)), tf.nn.softplus(tf.Variable(np.sqrt(self.std_llib) * tf.ones(self.l_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		if self.zero_inflation:
			inference = ed.ReparameterizationKLqp({self.z_test: self.qz_test, self.lam_test: self.qlam_test, self.l_test: self.ql_test}, 
				data={self.likelihood_test: tf.cast(X, tf.float32), self.W0: self.qW0, self.W1: self.qW1, self.r: self.qr})
		else:
			inference = ed.ReparameterizationKLqp({self.z_test: self.qz_test, self.lam_test: self.qlam_test, self.l_test: self.ql_test}, 
				data={self.likelihood_test: tf.cast(X, tf.float32), self.W0: self.qW0, self.r: self.qr})

		return inference

	# def run_inference(self, inference, inference_val=None, n_iterations=1000):
	# 	N = self.l.shape.as_list()[0]
	# 	self.inference_e.initialize(n_iter=n_iterations, n_samples=self.n_mc_samples, optimizer=self.optimizer)
	# 	self.inference_m.initialize(optimizer=self.optimizer)

	# 	self.sess = ed.get_session()
	# 	tf.global_variables_initializer().run()

	# 	for i in range(self.inference_e.n_iter):
	# 		info_dict = self.inference_e.update()
	# 		self.inference_m.update()
	# 		info_dict['loss'] = info_dict['loss'] / N
	# 		self.inference_e.print_progress(info_dict)
	# 		self.loss_dict['t_loss'].append(info_dict["loss"])

	# 	self.inference_e.finalize()
	# 	self.inference_m.finalize()

	def run_inference(self, inference, inference_val=None, n_iterations=1000):
		N = self.l.shape.as_list()[0]
		inference.initialize(n_iter=n_iterations, n_samples=self.n_mc_samples, optimizer=self.optimizer)

		if inference_val is not None:
			N_val= self.l_test.shape.as_list()[0]
			inference_val.initialize(n_samples=self.n_mc_samples, optimizer=self.optimizer)

		self.sess = ed.get_session()
		tf.global_variables_initializer().run()

		for i in range(inference.n_iter):
			info_dict = inference.update()
			info_dict['loss'] = info_dict['loss'] / N
			inference.print_progress(info_dict)
			self.loss_dict['t_loss'].append(info_dict["loss"])

			if inference_val is not None:
				self.sess.run(inference_val.reset)
				self.sess.run(tf.variables_initializer(self.ql_test.get_variables()))
				self.sess.run(tf.variables_initializer(self.qz_test.get_variables()))
				self.sess.run(tf.variables_initializer(self.qlam_test.get_variables()))
				for _ in range(5):
					val_info_dict = inference_val.update()
				self.loss_dict['v_loss'].append(val_info_dict['loss'] / N_val)

		inference.finalize()

		if inference_val is not None:
			inference_val.finalize()

	def evaluate_loglikelihood(self, X, batch_idx=None):
		"""
		This is the ELBO, which is a lower bound on the marginal log-likelihood.
		We perform some local optimization on the new data points to obtain the ELBO of the new data.
		"""
		N = X.shape[0]
		P = X.shape[1]
		K = self.n_components

		# Define new graph conditioned on the posterior global factors
		z_test = Gamma(2. * tf.ones([N, K]), 1. * tf.ones([N, K]))
		l_test = TransformedDistribution(
		    distribution=Normal(self.mean_llib * tf.ones([N,1]), np.sqrt(self.std_llib) * tf.ones([N,1])),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		if batch_idx is not None and self.n_batches > 0:
			rho_test = tf.matmul(tf.concat([z_test, tf.cast(tf.one_hot(batch_idx[:, 0], self.n_batches), tf.float32)], axis=1), self.W0)
		else:
			rho_test = tf.matmul(z_test, self.W0)
		rho_test = rho_test / tf.reshape(tf.reduce_sum(rho_test, axis=1), (-1, 1)) # NxP

		lam_test = Gamma(self.r, self.r / (rho_test * l_test))

		if self.zero_inflation:
			if batch_idx is not None and self.n_batches > 0:
				logit_pi_test = tf.matmul(tf.concat([z_test, tf.cast(tf.one_hot(batch_idx[:, 0], self.n_batches), tf.float32)], axis=1), self.W1)
			else:
				logit_pi_test = tf.matmul(z_test, self.W1)

			pi_test = tf.minimum(tf.maximum(tf.nn.sigmoid(logit_pi_test), 1e-7), 1.-1e-7)
			cat_test = Categorical(probs=tf.stack([pi_test, 1.-pi_test], axis=2))

			components_test = [
			    Poisson(rate=1e-30 * tf.ones([N, P])),
			    Poisson(rate=lam_test)
			]
			likelihood_test = Mixture(cat=cat_test, components=components_test)
		else:
			likelihood_test = Poisson(rate=lam_test)

		qz_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(z_test.shape)), tf.nn.softplus(tf.Variable(1. * tf.ones(z_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		qlam_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(lam_test.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(lam_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		ql_test = TransformedDistribution(
		    distribution=Normal(tf.Variable(self.mean_llib * tf.ones(l_test.shape)), tf.nn.softplus(tf.Variable(np.sqrt(self.std_llib) * tf.ones(l_test.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		if self.zero_inflation:
			inference_local = ed.ReparameterizationKLqp({z_test: qz_test, lam_test: qlam_test, l_test: ql_test}, 
				data={likelihood_test: tf.cast(X, tf.float32), self.W0: self.est_qW0, self.W1: self.est_qW1, self.r: self.est_qr})
		else:
			inference_local = ed.ReparameterizationKLqp({z_test: qz_test, lam_test: qlam_test, l_test: ql_test}, 
				data={likelihood_test: tf.cast(X, tf.float32), self.W0: self.est_qW0, self.r: self.est_qr})
		
		inference_local.run(n_iter=self.test_iterations, n_samples=self.n_mc_samples)

		return -self.sess.run(inference_local.loss, feed_dict={likelihood_test: X.astype('float32')}) / N

	def posterior_nb_mean(self):
		est_rho = self.sess.run(self.rho, feed_dict={self.z: np.exp(self.qz.distribution.loc.eval()), self.W0: np.exp(self.qW0.distribution.loc.eval())})
		est_l = 1
		if self.scalings:
			est_l = np.exp(self.ql.distribution.loc.eval())
		est_mean = est_rho * est_l

		return est_mean

	def define_stochastic_model(self, P, K):
		M = self.minibatch_size

		self.W0 = Gamma(0.1 * tf.ones([K, P]), 0.3 * tf.ones([K, P]))
		if self.zero_inflation:
			self.W1 = Normal(tf.zeros([K, P]), tf.ones([K, P]))

		self.z = Gamma(16. * tf.ones([M, K]), 4. * tf.ones([M, K]))

		self.r = Gamma(2. * tf.ones([P,]), 1. * tf.ones([P,]))

		self.l = TransformedDistribution(
		    distribution=Normal(self.mean_llib * tf.ones([M,1]), np.sqrt(self.std_llib) * tf.ones([M,1])),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		self.rho = tf.matmul(self.z, self.W0)
		self.rho = self.rho / tf.reshape(tf.reduce_sum(self.rho, axis=1), (-1, 1)) # NxP

		self.lam = Gamma(self.r, self.r / (self.rho * self.l))

		if self.zero_inflation:
			self.logit_pi = tf.matmul(self.z, self.W1)
			self.pi = tf.minimum(tf.maximum(tf.nn.sigmoid(self.logit_pi), 1e-7), 1.-1e-7)
			
			self.cat = Categorical(probs=tf.stack([self.pi, 1.-self.pi], axis=2))

			self.components = [
			    Poisson(rate=1e-30*tf.ones([M, P])),
			    Poisson(rate=self.lam)
			]

			self.likelihood = Mixture(cat=self.cat, components=self.components)
		else:
			self.likelihood = Poisson(rate=self.lam)

	def define_stochastic_inference(self, N, P, K):
		M = self.minibatch_size

		qz_vars = [tf.Variable(tf.ones([N, K]), name='qz_loc'), tf.Variable(1. * tf.ones([N, K]), name='qz_scale')]
		qlam_vars = [tf.Variable(tf.ones([N, P]), name='qlam_loc'), tf.Variable(0.01 * tf.ones([N, P]), name='qlam_scale')]
		ql_vars = [tf.Variable(mean_llib * tf.ones([N, 1]), name='ql_loc'), 
		           tf.Variable(np.sqrt(var_llib) * tf.ones([N, 1]), name='ql_scale')]
		qlocal_vars = [qz_vars, qlam_vars, ql_vars]
		qlocal_vars = [item for sublist in qlocal_vars for item in sublist]

		idx_ph = tf.placeholder(tf.int32, M)
		self.qz = TransformedDistribution(
		    distribution=Normal(tf.gather(qlocal_vars[0], idx_ph), tf.nn.softplus(tf.gather(qlocal_vars[1], idx_ph))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		self.qlam = TransformedDistribution(
		    distribution=Normal(tf.gather(qlocal_vars[2], idx_ph), tf.nn.softplus(tf.gather(qlocal_vars[3], idx_ph))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		self.ql = TransformedDistribution(
		    distribution=Normal(tf.gather(qlocal_vars[4], idx_ph), tf.nn.softplus(tf.gather(qlocal_vars[5], idx_ph))),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		self.qW0 = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.W0.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.W0.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())
		self.qW1 = Normal(tf.Variable(tf.zeros(self.W1.shape)), tf.nn.softplus(tf.Variable(tf.ones(self.W1.shape))))
		self.qr = TransformedDistribution(
		    distribution=Normal(tf.Variable(tf.ones(self.r.shape)), tf.nn.softplus(tf.Variable(0.01 * tf.ones(self.r.shape)))),
		    bijector=tf.contrib.distributions.bijectors.Exp())

		x_ph = tf.placeholder(tf.float32, [M, P])
		inference_global = ed.ReparameterizationKLqp({self.r: qr, self.W0: qW0, self.W1: qW1}, data={self.likelihood: x_ph, self.z: qz, self.lam: qlam, self.al: ql})
		inference_local = ed.ReparameterizationKLqp({z: qz, lam: qlam, l: ql}, data={likelihood: x_ph, r: qr, W0: qW0, W1: qW1})

	def run_stochastic_inference(self, inference_local, inference_global, N, n_iterations=100):
		M = self.minibatch_size

		# Run inference
		inference_global.initialize(scale={self.likelihood: float(N) / M, self.z: float(N) / M, self.lam: float(N) / M, self.l: float(N) / M})
		inference_local.initialize(scale={self.likelihood: float(N) / M, self.z: float(N) / M, self.lam: float(N) / M, self.l: float(N) / M})

		tf.global_variables_initializer().run()

		loss = []
     
		n_iter_per_epoch = N // M
		pbar = ed.Progbar(n_epochs)
		for epoch in range(n_epochs):
		#     print("Epoch: {0}".format(epoch))
		    avg_loss = 0.0

		    for t in range(1, n_iter_per_epoch + 1):
		        x_batch, idx_batch = next_batch(X_train, M)
		        
		#         inference_local.update(feed_dict={x_ph: x_batch})
		        for _ in range(5):  # make local inferences
		            info_dict = inference_local.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})

		        info_dict = inference_global.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch})
		        avg_loss += info_dict['loss']
		        
		    # Print a lower bound to the average marginal likelihood for a cell
		    avg_loss /= n_iter_per_epoch
		    avg_loss /= M

		#     print("-log p(x) <= {:0.3f}\n".format(avg_loss), end='\r')
		    loss.append(avg_loss)
		    pbar.update(epoch, values={'Loss': avg_loss})
		    
		inference_global.finalize()
		inference_local.finalize()

def lognormal_q(shape, name=None):
  with tf.variable_scope(name, default_name="lognormal_q"):
    min_scale = 1e-5
    loc = tf.get_variable(
    	"loc", shape, initializer=tf.random_normal_initializer(mean=1., stddev=0.01))
    scale = tf.get_variable(
        "scale", shape, initializer=tf.random_normal_initializer(mean=1., stddev=0.01))
    rv = TransformedDistribution(
        distribution=Normal(loc, tf.maximum(tf.nn.softplus(scale), min_scale)),
        bijector=tf.contrib.distributions.bijectors.Exp())
    return rv

def next_batch(x_train, M):
    idx_batch = np.random.choice(x_train.shape[0], M)
    return x_train[idx_batch, :], idx_batch