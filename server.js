const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const NAME = "PRITESH AI PREDICTOR ULTRA";
const API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json";

// ========== ADVANCED MULTI-MODEL PREDICTION ENGINE ==========

/**
 * Markov Chain Model - Analyzes transition probabilities between consecutive outcomes
 * Higher-order Markov captures patterns of length 2 and 3
 */
class MarkovChainModel {
    constructor(order = 2) {
        this.order = order;
        this.transitions = new Map();
        this.totalTransitions = 0;
    }

    update(history) {
        if (history.length <= this.order) return;
        
        for (let i = this.order; i < history.length; i++) {
            // Create state from previous 'order' outcomes
            const state = history.slice(i - this.order, i).join(',');
            const next = history[i];
            
            if (!this.transitions.has(state)) {
                this.transitions.set(state, new Map());
            }
            const nextMap = this.transitions.get(state);
            nextMap.set(next, (nextMap.get(next) || 0) + 1);
            this.totalTransitions++;
        }
    }

    predict(history) {
        if (history.length < this.order) return null;
        
        // Get the last 'order' outcomes as current state
        const currentState = history.slice(-this.order).join(',');
        const nextMap = this.transitions.get(currentState);
        
        if (!nextMap) return null;
        
        // Find most probable next outcome
        let maxCount = 0;
        let predicted = null;
        for (const [outcome, count] of nextMap) {
            if (count > maxCount) {
                maxCount = count;
                predicted = outcome;
            }
        }
        
        return predicted !== null ? parseInt(predicted) : null;
    }
}

/**
 * Frequency Analysis Model - Tracks hot/cold numbers with weighted recency
 * More recent draws have higher weight for better accuracy
 */
class FrequencyModel {
    constructor(windowSize = 20) {
        this.windowSize = windowSize;
        this.frequencies = new Array(10).fill(0);
        this.weightedFrequencies = new Array(10).fill(0);
    }

    update(history) {
        // Reset frequencies
        this.frequencies.fill(0);
        this.weightedFrequencies.fill(0);
        
        // Use only recent history within window
        const recentHistory = history.slice(-this.windowSize);
        
        recentHistory.forEach((outcome, idx) => {
            // Weight: more recent = higher weight (exponential decay)
            const weight = Math.exp(-(recentHistory.length - 1 - idx) / 5);
            this.frequencies[outcome]++;
            this.weightedFrequencies[outcome] += weight;
        });
    }

    predict() {
        // Find outcome with highest weighted frequency
        let maxWeight = -1;
        let predicted = 0;
        for (let i = 0; i < 10; i++) {
            if (this.weightedFrequencies[i] > maxWeight) {
                maxWeight = this.weightedFrequencies[i];
                predicted = i;
            }
        }
        return predicted;
    }
    
    getHotNumbers() {
        const threshold = this.frequencies.reduce((a,b) => a+b, 0) / 10;
        return this.frequencies.map((f, i) => ({ number: i, frequency: f }))
            .filter(item => item.frequency > threshold)
            .sort((a,b) => b.frequency - a.frequency)
            .map(item => item.number);
    }
    
    getColdNumbers() {
        const threshold = this.frequencies.reduce((a,b) => a+b, 0) / 10;
        return this.frequencies.map((f, i) => ({ number: i, frequency: f }))
            .filter(item => item.frequency < threshold)
            .sort((a,b) => a.frequency - b.frequency)
            .map(item => item.number);
    }
}

/**
 * Pattern Recognition Model - Detects repeating sequences and cycles
 * Uses FFT-inspired cycle detection and gap analysis
 */
class PatternRecognitionModel {
    constructor() {
        this.patternCache = new Map();
        this.cycleLengths = new Array(10).fill(0);
    }

    update(history) {
        // Detect cycles for each outcome
        for (let outcome = 0; outcome < 10; outcome++) {
            const positions = [];
            history.forEach((val, idx) => {
                if (val === outcome) positions.push(idx);
            });
            
            if (positions.length >= 3) {
                const gaps = [];
                for (let i = 1; i < positions.length; i++) {
                    gaps.push(positions[i] - positions[i-1]);
                }
                // Average gap as cycle length
                const avgGap = gaps.reduce((a,b) => a+b, 0) / gaps.length;
                this.cycleLengths[outcome] = Math.round(avgGap);
            }
        }
        
        // Detect repeating patterns of length 2 and 3
        for (let length = 2; length <= 3; length++) {
            for (let i = 0; i <= history.length - length; i++) {
                const pattern = history.slice(i, i + length).join(',');
                const next = history[i + length];
                
                if (!this.patternCache.has(pattern)) {
                    this.patternCache.set(pattern, new Map());
                }
                const nextMap = this.patternCache.get(pattern);
                nextMap.set(next, (nextMap.get(next) || 0) + 1);
            }
        }
    }

    predict(history) {
        // Try pattern matching for length 2 and 3
        for (let length = 3; length >= 2; length--) {
            if (history.length >= length) {
                const pattern = history.slice(-length).join(',');
                const nextMap = this.patternCache.get(pattern);
                
                if (nextMap && nextMap.size > 0) {
                    let maxCount = 0;
                    let predicted = null;
                    for (const [outcome, count] of nextMap) {
                        if (count > maxCount) {
                            maxCount = count;
                            predicted = parseInt(outcome);
                        }
                    }
                    if (predicted !== null) return predicted;
                }
            }
        }
        
        // If no pattern found, use cycle detection
        const lastOutcome = history[history.length - 1];
        if (this.cycleLengths[lastOutcome] > 0) {
            // Predict based on cycle
            return lastOutcome;
        }
        
        return null;
    }
}

/**
 * LSTM-Inspired Sequence Memory Model
 * Simulates LSTM behavior using sliding window with weighted memory
 */
class SequenceMemoryModel {
    constructor(windowSize = 10) {
        this.windowSize = windowSize;
        this.memoryWeights = new Array(windowSize).fill().map((_, i) => Math.exp(-i / 3));
        this.patternMemory = new Map();
    }

    update(history) {
        if (history.length < this.windowSize) return;
        
        // Store weighted patterns
        for (let i = this.windowSize; i <= history.length; i++) {
            const window = history.slice(i - this.windowSize, i);
            const next = history[i];
            
            if (next !== undefined) {
                const key = window.join(',');
                if (!this.patternMemory.has(key)) {
                    this.patternMemory.set(key, new Map());
                }
                const nextMap = this.patternMemory.get(key);
                nextMap.set(next, (nextMap.get(next) || 0) + 1);
            }
        }
    }

    predict(history) {
        if (history.length < this.windowSize) return null;
        
        const recentWindow = history.slice(-this.windowSize);
        const key = recentWindow.join(',');
        const nextMap = this.patternMemory.get(key);
        
        if (!nextMap || nextMap.size === 0) return null;
        
        let maxCount = 0;
        let predicted = null;
        for (const [outcome, count] of nextMap) {
            if (count > maxCount) {
                maxCount = count;
                predicted = parseInt(outcome);
            }
        }
        return predicted;
    }
}

/**
 * Anomaly Detection Model - Identifies unusual patterns that may signal reversals
 * Based on statistical deviation from expected behavior
 */
class AnomalyDetectionModel {
    constructor() {
        this.expectedProbabilities = new Array(10).fill(0.1);
        this.deviationThreshold = 2.0;
    }

    update(history) {
        // Calculate actual frequencies
        const actualFrequencies = new Array(10).fill(0);
        history.forEach(outcome => actualFrequencies[outcome]++);
        const total = history.length;
        
        // Calculate chi-square deviation
        let chiSquare = 0;
        for (let i = 0; i < 10; i++) {
            const expected = total * 0.1;
            const actual = actualFrequencies[i];
            if (expected > 0) {
                chiSquare += Math.pow(actual - expected, 2) / expected;
            }
        }
        
        // If distribution is too skewed, expect reversal to mean
        const isAnomalous = chiSquare > 15.0; // Critical value for 9 degrees of freedom
        
        // Identify overrepresented and underrepresented numbers
        const overrepresented = [];
        const underrepresented = [];
        for (let i = 0; i < 10; i++) {
            const actual = actualFrequencies[i];
            const expected = total * 0.1;
            const deviation = (actual - expected) / Math.sqrt(expected);
            
            if (deviation > this.deviationThreshold) {
                overrepresented.push(i);
            } else if (deviation < -this.deviationThreshold) {
                underrepresented.push(i);
            }
        }
        
        return { isAnomalous, overrepresented, underrepresented };
    }

    predict(analysis) {
        if (!analysis.isAnomalous) return null;
        
        // If distribution is anomalous, predict an underrepresented number (mean reversion)
        if (analysis.underrepresented.length > 0) {
            return analysis.underrepresented[Math.floor(Math.random() * analysis.underrepresented.length)];
        }
        return null;
    }
}

/**
 * Gradient Boosting-Inspired Weighted Voting Model
 * Combines predictions from all models with dynamic weights based on historical accuracy
 */
class GradientBoostingModel {
    constructor() {
        this.modelWeights = {
            markov: 0.25,
            frequency: 0.20,
            pattern: 0.20,
            sequence: 0.15,
            anomaly: 0.10,
            adaptive: 0.10
        };
        this.modelPerformance = {
            markov: { correct: 0, total: 0 },
            frequency: { correct: 0, total: 0 },
            pattern: { correct: 0, total: 0 },
            sequence: { correct: 0, total: 0 },
            anomaly: { correct: 0, total: 0 },
            adaptive: { correct: 0, total: 0 }
        };
        this.predictionHistory = [];
    }

    updateWeights() {
        // Calculate accuracy for each model
        const accuracies = {};
        let totalAccuracy = 0;
        
        for (const [model, perf] of Object.entries(this.modelPerformance)) {
            const accuracy = perf.total > 0 ? perf.correct / perf.total : 0.5;
            accuracies[model] = accuracy;
            totalAccuracy += accuracy;
        }
        
        // Update weights based on recent performance
        for (const model of Object.keys(this.modelWeights)) {
            if (totalAccuracy > 0) {
                this.modelWeights[model] = accuracies[model] / totalAccuracy;
            }
        }
        
        // Normalize weights
        const sum = Object.values(this.modelWeights).reduce((a,b) => a+b, 0);
        for (const model of Object.keys(this.modelWeights)) {
            this.modelWeights[model] /= sum;
        }
    }

    recordPrediction(model, wasCorrect) {
        if (this.modelPerformance[model]) {
            this.modelPerformance[model].total++;
            if (wasCorrect) this.modelPerformance[model].correct++;
        }
        // Update weights every 10 predictions
        if (Object.values(this.modelPerformance).reduce((s, p) => s + p.total, 0) % 10 === 0) {
            this.updateWeights();
        }
    }

    predict(predictions) {
        // Weighted voting
        const voteCount = new Array(10).fill(0);
        
        for (const [model, prediction] of Object.entries(predictions)) {
            if (prediction !== null && prediction >= 0 && prediction <= 9) {
                const weight = this.modelWeights[model] || 0.1;
                voteCount[prediction] += weight;
            }
        }
        
        // Find highest voted outcome
        let maxVotes = -1;
        let predicted = 0;
        for (let i = 0; i < 10; i++) {
            if (voteCount[i] > maxVotes) {
                maxVotes = voteCount[i];
                predicted = i;
            }
        }
        
        // Calculate confidence based on vote distribution
        const totalVotes = voteCount.reduce((a,b) => a+b, 0);
        const confidence = totalVotes > 0 ? (maxVotes / totalVotes) * 100 : 50;
        
        return { predicted, confidence, voteDistribution: voteCount };
    }
}

/**
 * Bayesian Updating Model - Updates beliefs based on evidence
 * Implements naive Bayesian inference for probability recalculation
 */
class BayesianModel {
    constructor() {
        this.priorProbabilities = new Array(10).fill(0.1);
        this.likelihoods = new Array(10).fill().map(() => new Array(10).fill(0.1));
        this.evidenceCount = 0;
    }

    update(history) {
        if (history.length < 2) return;
        
        // Update likelihoods based on transitions
        for (let i = 1; i < history.length; i++) {
            const prev = history[i-1];
            const curr = history[i];
            this.likelihoods[prev][curr]++;
            this.evidenceCount++;
        }
        
        // Normalize likelihoods
        for (let i = 0; i < 10; i++) {
            const rowSum = this.likelihoods[i].reduce((a,b) => a+b, 0);
            if (rowSum > 0) {
                for (let j = 0; j < 10; j++) {
                    this.likelihoods[i][j] /= rowSum;
                }
            }
        }
        
        // Update prior probabilities based on recent frequencies
        const recentHistory = history.slice(-20);
        const frequencies = new Array(10).fill(0);
        recentHistory.forEach(outcome => frequencies[outcome]++);
        const total = recentHistory.length;
        for (let i = 0; i < 10; i++) {
            this.priorProbabilities[i] = total > 0 ? frequencies[i] / total : 0.1;
        }
    }

    predict(lastOutcome) {
        // Calculate posterior probabilities P(next | last) ∝ P(last → next) * P(next)
        const posterior = new Array(10).fill(0);
        for (let i = 0; i < 10; i++) {
            posterior[i] = this.likelihoods[lastOutcome][i] * this.priorProbabilities[i];
        }
        
        // Find maximum posterior
        let maxPosterior = -1;
        let predicted = 0;
        for (let i = 0; i < 10; i++) {
            if (posterior[i] > maxPosterior) {
                maxPosterior = posterior[i];
                predicted = i;
            }
        }
        
        return predicted;
    }
}

// ========== MAIN PREDICTOR CLASS ==========
class UltraPredictor {
    constructor() {
        this.markovChain = new MarkovChainModel(2);
        this.frequencyModel = new FrequencyModel(20);
        this.patternModel = new PatternRecognitionModel();
        this.sequenceModel = new SequenceMemoryModel(8);
        this.anomalyModel = new AnomalyDetectionModel();
        this.gradientModel = new GradientBoostingModel();
        this.bayesianModel = new BayesianModel();
        
        this.history = [];           // Raw number history (0-9)
        this.categoryHistory = [];   // BIG=1, SMALL=0
        this.predictionLog = [];      // Track predictions for performance evaluation
        this.performanceMetrics = {
            markov: { correct: 0, total: 0 },
            frequency: { correct: 0, total: 0 },
            pattern: { correct: 0, total: 0 },
            sequence: { correct: 0, total: 0 },
            anomaly: { correct: 0, total: 0 },
            bayesian: { correct: 0, total: 0 }
        };
    }

    updateHistory(outcome) {
        this.history.push(outcome);
        if (this.history.length > 200) this.history.shift();
        
        const category = outcome >= 5 ? 1 : 0;
        this.categoryHistory.push(category);
        if (this.categoryHistory.length > 100) this.categoryHistory.shift();
        
        // Update all models with new data
        this.markovChain.update(this.history);
        this.frequencyModel.update(this.history);
        this.patternModel.update(this.history);
        this.sequenceModel.update(this.history);
        this.anomalyModel.update(this.history);
        this.bayesianModel.update(this.history);
    }

    generatePredictions() {
        if (this.history.length < 5) {
            return {
                markov: null,
                frequency: null,
                pattern: null,
                sequence: null,
                anomaly: null,
                bayesian: null,
                adaptive: null
            };
        }
        
        const predictions = {
            markov: this.markovChain.predict(this.history),
            frequency: this.frequencyModel.predict(),
            pattern: this.patternModel.predict(this.history),
            sequence: this.sequenceModel.predict(this.history),
            bayesian: this.bayesianModel.predict(this.history[this.history.length - 1])
        };
        
        // Anomaly prediction
        const anomalyAnalysis = this.anomalyModel.update(this.history);
        predictions.anomaly = this.anomalyModel.predict(anomalyAnalysis);
        
        // Adaptive prediction based on recent trend
        predictions.adaptive = this.getAdaptivePrediction();
        
        return predictions;
    }

    getAdaptivePrediction() {
        if (this.history.length < 10) return null;
        
        // Analyze recent trend
        const recent = this.history.slice(-10);
        const recentAvg = recent.reduce((a,b) => a+b, 0) / 10;
        const overallAvg = this.history.reduce((a,b) => a+b, 0) / this.history.length;
        
        // If recent average is significantly different from overall, expect regression
        if (Math.abs(recentAvg - overallAvg) > 2.0) {
            return Math.round(overallAvg);
        }
        
        // Otherwise, follow recent momentum
        const lastFew = this.history.slice(-3);
        const momentum = (lastFew[lastFew.length-1] - lastFew[0]) / 2;
        const prediction = Math.round(this.history[this.history.length-1] + momentum);
        return Math.max(0, Math.min(9, prediction));
    }

    getFinalPrediction() {
        const predictions = this.generatePredictions();
        const result = this.gradientModel.predict(predictions);
        
        // Store prediction for later performance tracking
        this.predictionLog.push({
            timestamp: Date.now(),
            predictions: predictions,
            finalPrediction: result.predicted,
            confidence: result.confidence
        });
        
        // Keep only last 100 predictions
        if (this.predictionLog.length > 100) this.predictionLog.shift();
        
        return result;
    }

    evaluatePrediction(actualOutcome, predictedOutcome) {
        const isCorrect = (actualOutcome >= 5) === (predictedOutcome >= 5);
        
        // Update performance metrics for each model
        const lastPrediction = this.predictionLog[this.predictionLog.length - 1];
        if (lastPrediction) {
            for (const [model, pred] of Object.ent
