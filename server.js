const express = require('express');
const axios = require('axios');
const NodeCache = require('node-cache');

const app = express();
const PORT = process.env.PORT || 3000;

const NAME = "ADVANCED WINGO PREDICTOR V4.0";
const API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json";

// Cache for API responses
const cache = new NodeCache({ stdTTL: 30 });

// ========== ADVANCED STATISTICAL MODELS ==========

// 1. Markov Chain Model with Memory 5
class MarkovChainModel {
    constructor() {
        this.transitions = new Map();
        this.order = 5;
        this.initialized = false;
    }

    update(sequence) {
        for (let i = 0; i < sequence.length - this.order; i++) {
            const state = sequence.slice(i, i + this.order).join('');
            const next = sequence[i + this.order];
            if (!this.transitions.has(state)) {
                this.transitions.set(state, new Map());
            }
            const nextStates = this.transitions.get(state);
            nextStates.set(next, (nextStates.get(next) || 0) + 1);
        }
        this.initialized = true;
    }

    predict(lastStates) {
        if (!this.initialized || lastStates.length < this.order) return 0.5;
        const state = lastStates.slice(-this.order).join('');
        const nextStates = this.transitions.get(state);
        if (!nextStates) return 0.5;
        
        let total = 0;
        let bigCount = 0;
        for (const [num, count] of nextStates) {
            total += count;
            if (num >= 5) bigCount += count;
        }
        return total > 0 ? bigCount / total : 0.5;
    }
}

// 2. Fourier Analysis for Cyclical Patterns
class FourierAnalyzer {
    constructor() {
        this.frequencies = new Array(12).fill(0);
        this.amplitudes = new Array(12).fill(0);
        this.phases = new Array(12).fill(0);
        this.initialized = false;
    }

    update(sequence) {
        const n = sequence.length;
        if (n < 24) return;
        
        // FFT approximation for dominant frequencies
        for (let k = 1; k <= 12; k++) {
            let real = 0, imag = 0;
            const freq = (2 * Math.PI * k) / n;
            for (let t = 0; t < n; t++) {
                const value = sequence[t] / 9.0; // Normalize to 0-1
                real += value * Math.cos(freq * t);
                imag += value * Math.sin(freq * t);
            }
            this.amplitudes[k-1] = Math.sqrt(real*real + imag*imag);
            this.phases[k-1] = Math.atan2(imag, real);
            this.frequencies[k-1] = k;
        }
        this.initialized = true;
    }

    predict(offset = 0) {
        if (!this.initialized) return 0.5;
        let prediction = 0;
        const n = 50; // Recent window size
        
        for (let k = 0; k < this.frequencies.length; k++) {
            const freq = (2 * Math.PI * this.frequencies[k]) / 50;
            prediction += this.amplitudes[k] * Math.cos(freq * (offset) + this.phases[k]);
        }
        // Normalize to 0-1 probability
        return Math.min(1, Math.max(0, prediction / this.amplitudes.reduce((a,b) => a+b, 0) + 0.5));
    }
}

// 3. Bayesian Network with Time Series
class BayesianTimeSeries {
    constructor() {
        this.means = [];
        this.variances = [];
        this.weights = [];
        this.windowSize = 20;
        this.initialized = false;
    }

    update(sequence) {
        const n = sequence.length;
        if (n < this.windowSize) return;
        
        // Calculate weighted moving statistics
        this.means = [];
        this.variances = [];
        this.weights = [];
        
        for (let lag = 1; lag <= 10; lag++) {
            const window = sequence.slice(-this.windowSize - lag, -lag);
            const mean = window.reduce((a,b) => a+b, 0) / window.length;
            const variance = window.reduce((a,b) => a + Math.pow(b - mean, 2), 0) / window.length;
            this.means.push(mean);
            this.variances.push(variance);
            this.weights.push(Math.exp(-lag / 5));
        }
        this.initialized = true;
    }

    predict(lastValue) {
        if (!this.initialized) return 0.5;
        
        let weightedPrediction = 0;
        let totalWeight = 0;
        
        for (let i = 0; i < this.means.length; i++) {
            const zScore = (lastValue - this.means[i]) / Math.sqrt(this.variances[i] + 1);
            const probability = 1 / (1 + Math.exp(-zScore));
            weightedPrediction += probability * this.weights[i];
            totalWeight += this.weights[i];
        }
        
        return weightedPrediction / totalWeight;
    }
}

// 4. LSTM-like Recurrent Pattern (Simulated)
class RecurrentPatternModel {
    constructor() {
        this.patterns = new Map();
        this.sequenceLength = 8;
        this.initialized = false;
    }

    update(sequence) {
        for (let i = 0; i < sequence.length - this.sequenceLength; i++) {
            const pattern = sequence.slice(i, i + this.sequenceLength).join('');
            const next = sequence[i + this.sequenceLength];
            if (!this.patterns.has(pattern)) {
                this.patterns.set(pattern, { big: 0, small: 0, numbers: new Map() });
            }
            const stats = this.patterns.get(pattern);
            if (next >= 5) stats.big++;
            else stats.small++;
            
            // Track specific number predictions
            if (!stats.numbers.has(next)) {
                stats.numbers.set(next, 0);
            }
            stats.numbers.set(next, stats.numbers.get(next) + 1);
        }
        this.initialized = true;
    }

    predict(pattern, predictNumber = false) {
        if (!this.initialized || !this.patterns.has(pattern)) {
            return predictNumber ? { probability: 0.5, number: null } : 0.5;
        }
        
        const stats = this.patterns.get(pattern);
        const total = stats.big + stats.small;
        
        if (predictNumber) {
            // Predict most probable single number
            let maxCount = 0;
            let predictedNumber = null;
            for (const [num, count] of stats.numbers) {
                if (count > maxCount) {
                    maxCount = count;
                    predictedNumber = num;
                }
            }
            return {
                probability: total > 0 ? stats.big / total : 0.5,
                number: predictedNumber
            };
        }
        
        return total > 0 ? stats.big / total : 0.5;
    }
}

// 5. Ensemble Predictor with Dynamic Weighting
class EnsemblePredictor {
    constructor() {
        this.models = {
            markov: new MarkovChainModel(),
            fourier: new FourierAnalyzer(),
            bayesian: new BayesianTimeSeries(),
            recurrent: new RecurrentPatternModel()
        };
        
        this.weights = {
            markov: 0.25,
            fourier: 0.25,
            bayesian: 0.25,
            recurrent: 0.25
        };
        
        this.performance = {
            markov: { wins: 0, total: 0 },
            fourier: { wins: 0, total: 0 },
            bayesian: { wins: 0, total: 0 },
            recurrent: { wins: 0, total: 0 }
        };
        
        this.history = [];
        this.numberHistory = [];
        this.binaryHistory = [];
    }

    update(actualNumber) {
        const actualBinary = actualNumber >= 5 ? 1 : 0;
        this.numberHistory.push(actualNumber);
        this.binaryHistory.push(actualBinary);
        
        // Keep history manageable
        if (this.numberHistory.length > 200) {
            this.numberHistory.shift();
            this.binaryHistory.shift();
        }
        
        // Update all models
        if (this.binaryHistory.length > 10) {
            this.models.markov.update(this.binaryHistory);
            this.models.fourier.update(this.numberHistory);
            this.models.bayesian.update(this.numberHistory);
            this.models.recurrent.update(this.binaryHistory);
        }
    }

    predict(singleNumber = false) {
        const predictions = {};
        const pattern = this.binaryHistory.slice(-8).join('');
        
        // Get predictions from each model
        if (this.binaryHistory.length >= 5) {
            predictions.markov = this.models.markov.predict(this.binaryHistory);
        }
        
        if (this.numberHistory.length >= 24) {
            predictions.fourier = this.models.fourier.predict(this.numberHistory.length);
        }
        
        if (this.numberHistory.length >= 20) {
            predictions.bayesian = this.models.bayesian.predict(
                this.numberHistory[this.numberHistory.length - 1] || 0
            );
        }
        
        if (this.binaryHistory.length >= 8) {
            const recurrentPred = this.models.recurrent.predict(pattern, singleNumber);
            predictions.recurrent = singleNumber ? recurrentPred.probability : recurrentPred;
            if (singleNumber) {
                predictions.recurrentNumber = recurrentPred.number;
            }
        }
        
        // Calculate weighted average
        let weightedSum = 0;
        let totalWeight = 0;
        let bestModel = null;
        let bestConfidence = 0;
        
        for (const [model, prediction] of Object.entries(predictions)) {
            if (typeof prediction === 'number') {
                const weight = this.weights[model] || 0.25;
                weightedSum += prediction * weight;
                totalWeight += weight;
                
                // Track best individual prediction
                const confidence = Math.abs(prediction - 0.5) * 2;
                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestModel = model;
                }
            }
        }
        
        const ensembleProb = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        const prediction = ensembleProb >= 0.5 ? "BIG" : "SMALL";
        const confidence = Math.abs(ensembleProb - 0.5) * 2;
        
        // Predict single number if requested
        let predictedNumber = null;
        if (singleNumber && predictions.recurrentNumber !== undefined) {
            // Use recurrent model's number prediction
            predictedNumber = predictions.recurrentNumber;
            
            // Cross-validate with ensemble probability
            if (predictedNumber !== null) {
                // Adjust based on ensemble prediction
                if (prediction === "BIG" && predictedNumber < 5) {
                    predictedNumber = Math.min(9, predictedNumber + 3);
                } else if (prediction === "SMALL" && predictedNumber >= 5) {
                    predictedNumber = Math.max(0, predictedNumber - 3);
                }
            }
        }
        
        return {
            prediction,
            confidence: (confidence * 100).toFixed(2) + '%',
            ensembleProb,
            modelScores: predictions,
            bestModel,
            predictedNumber: predictedNumber !== null ? predictedNumber : this.predictNumber(ensembleProb),
            details: {
                markovWeight: this.weights.markov,
                fourierWeight: this.weights.fourier,
                bayesianWeight: this.weights.bayesian,
                recurrentWeight: this.weights.recurrent
            }
        };
    }

    predictNumber(probability) {
        // Convert probability to number (0-9) with distribution
        // Higher probability = higher numbers
        const rawNumber = probability * 9;
        // Add some noise for realism
        const noise = (Math.random() - 0.5) * 1.5;
        let number = Math.round(rawNumber + noise);
        return Math.min(9, Math.max(0, number));
    }

    updateWeights(prediction, actualNumber) {
        const actualBinary = actualNumber >= 5 ? 1 : 0;
        const predictedBinary = prediction === "BIG" ? 1 : 0;
        const isCorrect = predictedBinary === actualBinary;
        
        // Update performance tracking
        for (const [model, score] of Object.entries(prediction.modelScores)) {
            if (typeof score === 'number') {
                this.performance[model].total++;
                const modelPrediction = score >= 0.5 ? 1 : 0;
                if (modelPrediction === actualBinary) {
                    this.performance[model].wins++;
                }
            }
        }
        
        // Dynamic weight adjustment based on performance
        if (this.performance.markov.total > 10) {
            const total = this.performance.markov.total;
            const newWeights = {};
            let weightSum = 0;
            
            for (const model of Object.keys(this.weights)) {
                const performance = this.performance[model];
                const accuracy = performance.total > 0 ? performance.wins / performance.total : 0.5;
                // Exponential weight based on accuracy
                newWeights[model] = Math.exp(accuracy * 3);
                weightSum += newWeights[model];
            }
            
            // Normalize weights
            for (const model of Object.keys(this.weights)) {
                this.weights[model] = newWeights[model] / weightSum;
            }
        }
    }
}

// ========== GLOBAL STATE ==========
const predictor = new EnsemblePredictor();
const resultsHistory = [];
let totalTrades = 0;
let wins = 0;
let lastProcessedPeriod = null;
let predictionsCache = new Map();
let isProcessing = false;

// ========== API FETCH FUNCTIONS ==========
async function fetchLatestResults(limit = 10) {
    try {
        const cacheKey = 'results_' + limit;
        const cached = cache.get(cacheKey);
        if (cached) return cached;
        
        const url = `${API_URL}?ts=${Date.now()}&limit=${limit}`;
        const response = await axios.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.ar-lottery01.com/",
                "Origin": "https://draw.ar-lottery01.com"
            },
            timeout: 15000
        });
        
        const data = response.data;
        let results = [];
        
        if (data?.data?.list) {
            results = data.data.list.map(item => ({
                period: String(item.issue || item.issueNumber),
                number: parseInt(item.number)
            }));
        } else if (data?.list) {
            results = data.list.map(item => ({
                period: String(item.issue || item.issueNumber),
                number: parseInt(item.number)
            }));
        }
        
        if (results.length > 0) {
            cache.set(cacheKey, results);
            return results;
        }
        
        return null;
    } catch (err) {
        console.error(`[API Error] ${err.message}`);
        return null;
    }
}

// ========== MAIN PREDICTION FLOW ==========
async function processPrediction() {
    if (isProcessing) return;
    isProcessing = true;
    
    try {
        console.log('\n' + '='.repeat(60));
        console.log(`📊 ${NAME} - Processing Period`);
        console.log('='.repeat(60));
        
        // Step 1: Fetch latest results
        console.log('[1] Fetching latest results...');
        const results = await fetchLatestResults(10);
        if (!results || results.length === 0) {
            console.log('[ERROR] No results fetched');
            isProcessing = false;
            return;
        }
        
        const currentResult = results[0];
        const period = currentResult.period;
        const number = currentResult.number;
        
        console.log(`[LIVE] Period: ${period} | Number: ${number}`);
        
        // Step 2: Check if we have a prediction for this period
        if (predictionsCache.has(period)) {
            console.log('[2] Checking previous prediction...');
            const prediction = predictionsCache.get(period);
            const actualCategory = number >= 5 ? "BIG" : "SMALL";
            const isWin = prediction.prediction === actualCategory;
            
            totalTrades++;
            if (isWin) wins++;
            
            // Update predictor with actual result
            predictor.update(number);
            
            // Update weights based on result
            predictor.updateWeights(prediction, number);
            
            // Record result
            const resultEntry = {
                period: period,
                prediction: prediction.prediction,
                actual: actualCategory,
                actualNumber: number,
                result: isWin ? "WIN ✅" : "LOSS ❌",
                confidence: prediction.confidence,
                predictedNumber: prediction.predictedNumber,
                modelScores: prediction.modelScores,
                timestamp: new Date().toISOString()
            };
            
            resultsHistory.unshift(resultEntry);
            if (resultsHistory.length > 20) resultsHistory.pop();
            
            console.log(`[RESULT] Period ${period} | Pred: ${prediction.prediction} (${prediction.predictedNumber}) | Actual: ${actualCategory} (${number}) | ${isWin ? '✅ WIN' : '❌ LOSS'}`);
            
            // Clear prediction
            predictionsCache.delete(period);
        } else {
            console.log('[2] No previous prediction found - adding to history...');
            predictor.update(number);
            
            // Also add previous results to train the model
            for (let i = 1; i < Math.min(results.length, 10); i++) {
                predictor.update(results[i].number);
            }
        }
        
        // Step 3: Predict next period
        console.log('[3] Generating prediction for next period...');
        const nextPeriod = String(parseInt(period) + 1);
        
        // Get prediction (including single number)
        const prediction = predictor.predict(true);
        
        // Store prediction
        predictionsCache.set(nextPeriod, prediction);
        
        console.log(`[PREDICTION] Next Period: ${nextPeriod}`);
        console.log(`  → BIG/SMALL: ${prediction.prediction} (${prediction.confidence})`);
        console.log(`  → Single Number: ${prediction.predictedNumber}`);
        console.log(`  → Ensemble Probability: ${(prediction.ensembleProb * 100).toFixed(2)}%`);
        console.log(`  → Best Model: ${prediction.bestModel || 'N/A'}`);
        
        if (prediction.modelScores) {
            console.log('  → Model Scores:');
            for (const [model, score] of Object.entries(prediction.modelScores)) {
                if (typeof score === 'number') {
                    console.log(`      ${model}: ${(score * 100).toFixed(2)}%`);
                }
            }
        }
        
        console.log(`  → Model Weights:`, prediction.details);
        
        // Step 4: Simulated bet execution (with 5-second delay)
        console.log('[4] Waiting 5 seconds before executing bet...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Step 5: Execute bet
        console.log('[5] Executing bet...');
        const betResult = await executeBet(
            nextPeriod,
            prediction.prediction,
            prediction.confidence,
            prediction.predictedNumber
        );
        
        console.log('[BET]', JSON.stringify(betResult, null, 2));
        
        // Display statistics
        const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
        console.log(`\n📈 STATISTICS:`);
        console.log(`  Total Trades: ${totalTrades}`);
        console.log(`  Wins: ${wins}`);
        console.log(`  Losses: ${totalTrades - wins}`);
        console.log(`  Win Rate: ${winRate}%`);
        console.log(`  Target: 85%`);
        console.log(`  Performance: ${winRate >= 85 ? '🎯 ON TARGET' : '📈 IMPROVING'}`);
        console.log('='.repeat(60) + '\n');
        
        lastProcessedPeriod = period;
        
    } catch (err) {
        console.error(`[ERROR] ${err.message}`);
    }
    
    isProcessing = false;
}

// ========== BET EXECUTION ==========
async function executeBet(period, prediction, confidence, number) {
    // Simulated bet execution
    // In production, this would call your betting platform API
    const betAmount = 10;
    const payoutMultiplier = 1.98;
    const winAmount = betAmount * payoutMultiplier;
    
    // Determine if this is a single number or BIG/SMALL bet
    const betType = number !== null ? 'SINGLE_NUMBER' : 'BIG_SMALL';
    const numberBet = number !== null ? number : null;
    
    return {
        success: true,
        betId: `BET-${Date.now()}`,
        period: period,
        prediction: prediction,
        predictedNumber: number,
        betType: betType,
        amount: betAmount,
        potentialWin: winAmount,
        confidence: confidence,
        timestamp: new Date().toISOString(),
        risk: parseFloat(confidence) >= 70 ? 'LOW' : 'MEDIUM',
        strategy: 'Ensemble AI with Statistical Models'
    };
}

// ========== START THE BOT ==========
async function startBot() {
    console.log(`🚀 ${NAME} Starting...`);
    console.log('📋 System Features:');
    console.log('  • 5 Advanced Statistical Models');
    console.log('  • Markov Chain (Memory 5)');
    console.log('  • Fourier Analysis (Cyclical)');
    console.log('  • Bayesian Time Series');
    console.log('  • Recurrent Pattern Recognition');
    console.log('  • Dynamic Ensemble Weighting');
    console.log('  • Single Number Prediction');
    console.log('  • 1-Minute Auto Trading\n');
    
    // Initial prediction
    await processPrediction();
    
    // Run every 60 seconds (1 minute)
    setInterval(async () => {
        await processPrediction();
    }, 60000);
}

// ========== EXPRESS ROUTES ==========
app.get('/trade', (req, res) => {
    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
    
    // Get latest predictions
    const predictions = Array.from(predictionsCache.entries()).map(([period, pred]) => ({
        period: period,
        prediction: pred.prediction,
        confidence: pred.confidence,
        predictedNumber: pred.predictedNumber,
        probability: pred.ensembleProb
    }));
    
    const latestPrediction = predictions.length > 0 ? predictions[0] : null;
    
    res.json({
        bot: {
            name: NAME,
            status: "active",
            version: "4.0",
            flow: "API Poll → Predict → 5s Delay → Execute Bet"
        },
        currentPrediction: latestPrediction || {
            period: "WAITING",
            prediction: "BIG",
            confidence: "50.00%",
            predictedNumber: 5
        },
        performance: {
            totalTrades: totalTrades,
            totalWins: wins,
            totalLosses: totalTrades - wins,
            winRate: `${winRate}%`,
            targetAccuracy: "85%",
            achieved: winRate >= 85 ? "✅" : "📈"
        },
        modelWeights: latestPrediction ? latestPrediction.modelWeights : null,
        last10Results: resultsHistory.slice(0, 10),
        timestamp: new Date().toISOString()
    });
});

app.get('/status', (req, res) => {
    res.json({
        status: "active",
        name: NAME,
        lastProcessedPeriod: lastProcessedPeriod,
        predictionsInCache: predictionsCache.size,
        totalTrades: totalTrades,
        winRate: totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) + '%' : '0%',
        isProcessing: isProcessing
    });
});

app.get('/history', (req, res) => {
    const limit = parseInt(req.query.limit) || 20;
    res.json({
        results: resultsHistory.slice(0, limit),
        total: resultsHistory.length
    });
});

app.get('/predict', async (req, res) => {
    // Manual prediction trigger for testing
    const prediction = predictor.predict(true);
    res.json({
        prediction: prediction,
        timestamp: new Date().toISOString()
    });
});

app.get('/', (req, res) => {
    res.json({
        status: "online",
        name: NAME,
        type: "Advanced Wingo Predictor",
        features: [
            "5 Statistical Models",
            "Dynamic Ensemble",
            "Single Number Prediction",
            "Real-time Trading",
            "Performance Tracking"
        ],
        endpoints: {
            trade: "/trade - Get current prediction and performance",
            status: "/status - Bot status",
            history: "/history - Result history",
            predict: "/predict - Manual prediction"
        }
    });
});

app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

// ========== START SERVER ==========
app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
    console.log(`📡 Trade API: http://localhost:${PORT}/trade`);
    console.log(`📊 Status: http://localhost:${PORT}/status\n`);
});

// Start the bot
startBot().catch(console.error);

module.exports = app;
