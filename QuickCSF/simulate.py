# -*- coding: utf-8 -*
'''Simulate a QuickCSF experiment'''

import logging
import time

import numpy

import matplotlib.pyplot as plt
import pathlib

from . import QuickCSF
from .plot import plot

# qw: import our utils
from utility import utils

logger = logging.getLogger('QuickCSF.simulate')

def runSimulation(
	trials=30,
	imagePath=None,
	usePerfectResponses=False,
	stimuli={
		'minContrast':0.01, 'maxContrast':1, 'contrastResolution':24,
		'minFrequency':.2, 'maxFrequency':36, 'frequencyResolution':20,
	},
	parameters={
		'truePeakSensitivity':18, 'truePeakFrequency':11,
		'trueBandwidth':12, 'trueDelta':11,
	},
	d=0.5,
	psiGamma=None,
	psiLambda=None,
	psiSigma=None,
	sigmoidType=None,
	trueThresholdCurve=None,
	randomSeed=None,
	timepoints=None,
	showPlots=False
):
	
	if trueThresholdCurve is not None:
		# qw: create labelling curve
		# different units to match mlcsf experiments
		labeling_curve = trueThresholdCurve.copy()
		labeling_curve[:, 0] = (numpy.log2(10) * trueThresholdCurve[:, 0]) - numpy.log2(.125)
		labeling_cs = utils.create_cubic_spline(labeling_curve)

	rmses = []
	times = []

	if timepoints is not None:
		startTime = time.perf_counter()
		timepoints_set = set(timepoints)

	# logger.info('Starting simulation')

	# qw: set random seed
	if randomSeed: numpy.random.seed(randomSeed)
	else: numpy.random.seed()

	if imagePath is not None:
		pathlib.Path(imagePath).mkdir(parents=True, exist_ok=True)

	stimulusSpace = [
		QuickCSF.makeContrastSpace(stimuli['minContrast'], stimuli['maxContrast'], stimuli['contrastResolution']),
		QuickCSF.makeFrequencySpace(stimuli['minFrequency'], stimuli['maxFrequency'], stimuli['frequencyResolution'])
	]

	# Transform these bounds - used to make grid
	x_min = utils.logFreq().forward(stimuli['minFrequency'])
	x_max = utils.logFreq().forward(stimuli['maxFrequency'])
	y_min = utils.logContrast().forward(stimuli['maxContrast'])  # max and min get flipped when inverting
	y_max = utils.logContrast().forward(stimuli['minContrast'])

	# Make grid
	xs, ys = stimuli['frequencyResolution'], stimuli['contrastResolution']
	_, xx, _ = utils.create_evaluation_grid(x_min, x_max, y_min, y_max, xs, ys)

	unmappedTrueParams = None

	# qw
	if parameters is not None:
		unmappedTrueParams = numpy.array([[
			parameters['truePeakSensitivity'],
			parameters['truePeakFrequency'],
			parameters['trueBandwidth'],
			parameters['trueDelta'],
		]])

	qcsf = QuickCSF.QuickCSFEstimator(stimulusSpace, d)

	graph = plot(qcsf, unmappedTrueParams=unmappedTrueParams, trueThresholdCurve=trueThresholdCurve, show=showPlots)

	# Trial loop
	for i in range(trials):
		# Get the next stimulus
		stimulus = qcsf.next()
		newStimValues = numpy.array([[stimulus.contrast, stimulus.frequency]])

		# Simulate a response
		if usePerfectResponses: # qw: not being used so no worries
			logger.debug('Simulating perfect response')
			frequency = newStimValues[:,1]
			trueSens = numpy.power(10, QuickCSF.csf_unmapped(unmappedTrueParams, numpy.array([frequency])))
			testContrast = newStimValues[:,0]
			testSens = 1 / testContrast

			response = trueSens > testSens
		else:
			logger.debug('Simulating human response')
			
			if parameters:
				p = qcsf._pmeas(unmappedTrueParams)
				response = numpy.random.rand() < p
			else:
				# qw: label points using our ground truth curve
				x1 = utils.logFreq().forward(stimulus.frequency)
				x2 = utils.logContrast().forward(stimulus.contrast)
				response = utils.simulate_labeling(x1, x2, labeling_cs, psiGamma, psiLambda, sigmoid_type=sigmoidType, psi_sigma=psiSigma)
		
		qcsf.markResponse(response)

		# Update the plot
		graph.clear()
		graph.set_title(f'Estimated Contrast Sensitivity Function ({i+1})')
		plot(qcsf, graph, unmappedTrueParams=unmappedTrueParams, trueThresholdCurve=trueThresholdCurve, show=showPlots)

		if imagePath is not None:
			plt.savefig(pathlib.Path(f'{imagePath}{i}-plot.png').resolve())
		
		# qw: calculate rmse for current qcsf
		params = qcsf.getResults()
		rmses.append(utils.getQcsfRMSE(
			xx=xx,
			cs=labeling_cs,
			peakSensitivity=params['peakSensitivity'],
			peakFrequency=params['peakFrequency'],
			logBandwidth=params['bandwidth'],
			delta=params['delta'],
			qcsf=QuickCSF.csf
			))

		# time it
		if timepoints and i+1 in timepoints_set:
			times.append(time.perf_counter() - startTime)


	# qw: print param estimates and return with rmses and times
	paramEstimates = qcsf.getResults()
	print(f'Estimates = {paramEstimates}')
	if parameters is not None:
		trueParams = QuickCSF.mapCSFParams(unmappedTrueParams, True).T
		print(f'\tActuals = {trueParams}')
	
	plt.ioff()
	if showPlots: plt.show()
	else:
		plt.close()

	
	return rmses, times, paramEstimates

def entropyPlot(qcsf):
	params = numpy.arange(qcsf.paramComboCount).reshape(-1, 1)
	stims = numpy.arange(qcsf.stimComboCount).reshape(-1,1)

	p = qcsf._pmeas(params, stims)

	pbar = sum(p)/len(params)
	hbar = sum(QuickCSF.entropy(p)) / len(params)
	gain = QuickCSF.entropy(pbar) - hbar


	gain = -gain.reshape(qcsf.stimulusRanges[::-1]).T

	fig = plt.figure()
	graph = fig.add_subplot(1, 1, 1)
	plt.imshow(gain, cmap='hot')

	plt.ioff()
	plt.show()