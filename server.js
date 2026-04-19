const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const NAME = "PRITESH AI PREDICTOR (ADVANCED)";
const API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json";

// ========== ADVANCED ENSEMBLE AI (85% ACCURACY TARGET) ==========

// ---------- Model 1: Enhanced Logistic Regression with 15 features ----------
class AdvancedLogisticRegression {
    constructor(lr = 0.05, nFeatures = 15) {
        this.lr = lr;
        this.weights = Array(nFeatures).fill().map(() => (Math.random() - 0.5) * 0.5);
        this.bias = (Math.random() - 0.5) * 0.5;
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    predict(features) {
        let z = this.bias;
        for (let i = 0; i < features.length; i++) {
            z += features[i] * this.weights[i];
        }
        return this.sigmoid(z);
    }

    update(actual, features) {
        const pred = this.predict(features);
        const error = actual - pred;
        for (let i = 0; i < features.length; i++) {
            this.weights[i] += this.lr * error * features[i];
        }
        this.bias += this.lr * error;
    }
}

// ---------- Model 2: Naive Bayes (simple but effective for binary) ----------
class NaiveBayes {
    constructor() {
        this.priorBig = 0.5;
        this.priorSmall = 0.5;
        // Conditional probabilities for each feature (discretized)
        this.condBig = [];
        this.condSmall = [];
        this.initialized = false;
    }

    discretize(value, bins = 5) {
        // Map continuous feature into 5 bins (0-4)
        const min = -1, max = 1;
        const normalized = (value - min) / (max - min);
        return Math.min(bins - 1, Math.floor(normalized * bins));
    }

    initialize(nFeatures) {
        for (let i = 0; i < nFeatures; i++) {
            this.condBig.push(Array(5).fill(0.5)); // Laplace smoothing
            this.condSmall.push(Array(5).fill(0.5));
        }
        this.initialized = true;
    }

    update(actualBinary, features, nFeatures) {
        if (!this.initialized) this.initialize(nFeatures);
        const classIdx = actualBinary === 1 ? 'condBig' : 'condSmall';
        const other = actualBinary === 1 ? 'condSmall' : 'condBig';
        for (let i = 0; i < features.length; i++) {
            const bin = this.discretize(features[i]);
            this[classIdx][i][bin] += 1;
            // Smooth other class slightly to avoid zero
            if (this[other][i][bin] < 0.1) this[other][i][bin] += 0.01;
        }
        // Update priors
        this.priorBig = (this.priorBig * 99 + actualBinary) / 100;
        this.priorSmall = 1 - this.priorBig;
    }

    predict(features) {
        if (!this.initialized) return 0.5;
        let logProbBig = Math.log(this.priorBig);
        let logProbSmall = Math.log(this.priorSmall);
        for (let i = 0; i < features.length; i++) {
            const bin = this.discretize(features[i]);
            const probBig = this.condBig[i][bin] / this.condBig[i].reduce((a,b)=>a+b,0);
            const probSmall = this.condSmall[i][bin] / this.condSmall[i].reduce((a,b)=>a+b,0);
            logProbBig += Math.log(probBig + 1e-9);
            logProbSmall += Math.log(probSmall + 1e-9);
        }
        const probBig = Math.exp(logProbBig) / (Math.exp(logProbBig) + Math.exp(logProbSmall));
        return probBig;
    }
}

// ---------- Model 3: Pattern Matching (Markov Chain with memory 3) ----------
class PatternMatcher {
    constructor() {
        this.patterns = new Map(); // key: last3 pattern string, value: {big: count, small: count}
    }

    update(historyBinary) {
        // historyBinary is array of 0/1, most recent last
        if (historyBinary.length < 4) return;
        const last3 = historyBinary.slice(-3).join('');
        const next = historyBinary[historyBinary.length-1];
        if (!this.patterns.has(last3)) {
            this.patterns.set(last3, {big: 0, small: 0});
        }
        const stats = this.patterns.get(last3);
        if (next === 1) stats.big++;
        else stats.small++;
    }

    predict(last3Pattern) {
        if (!this.patterns.has(last3Pattern)) return 0.5;
        const stats = this.patterns.get(last3Pattern);
        const total = stats.big + stats.small;
        if (total === 0) return 0.5;
        return stats.big / total;
    }
}

// ---------- Ensemble that combines all 3 models with adaptive weights ----------
class EnsemblePredictor {
    constructor() {
        this.model1 = new AdvancedLogisticRegression(0.05, 15);
        this.model2 = new NaiveBayes();
        this.model3 = new PatternMatcher();
        this.weights = { m1: 0.4, m2: 0.35, m3: 0.25 };
        this.performance = { m1: 0, m2: 0, m3: 0, total: 0 };
    }

    extractFeatures(historyNumbers, periodParity) {
        // historyNumbers: array of actual numbers (0-9) most recent last? We'll use last 20 binary values
        // Convert to binary (1 for BIG, 0 for SMALL) for most features
        const binary = historyNumbers.map(n => n >= 5 ? 1 : 0);
        const len = binary.length;
        if (len === 0) return Array(15).fill(0);
        
        const last = binary[len-1] || 0;
        const last3 = binary.slice(-3);
        const last5 = binary.slice(-5);
        const last10 = binary.slice(-10);
        
        // Basic stats
        const avg3 = last3.reduce((a,b)=>a+b,0)/3;
        const avg5 = last5.reduce((a,b)=>a+b,0)/5;
        const avg10 = last10.reduce((a,b)=>a+b,0)/10;
        const variance5 = last5.reduce((sum,val)=>sum + Math.pow(val-avg5,2),0)/5;
        
        // Streak of same outcome
        let streak = 1;
        for (let i=len-2; i>=0 && binary[i]===last; i--) streak++;
        const streakNorm = Math.min(streak/10, 1);
        
        // Trend: difference between recent averages
        const trend35 = avg3 - avg5;
        const trend510 = avg5 - avg10;
        
        // Volatility (std dev of last 5)
        const volatility = Math.sqrt(variance5);
        
        // Pattern of last 3 as integer 0-7
        const last3Pattern = (binary[len-3]||0)*4 + (binary[len-2]||0)*2 + (binary[len-1]||0);
        
        // Parity of next period (0 or 1)
        const parity = periodParity;
        
        // Big/Small ratio in last 10
        const bigRatio10 = avg10;
        
        // Momentum: change from last to previous
        const momentum = len>=2 ? last - binary[len-2] : 0;
        
        // Hour of day effect (simulated using period number mod 24)
        const hourEffect = (parseInt(periodParity) % 24) / 24; // dummy if period not passed
        
        // Return 15 features
        return [
            last, avg3, avg5, avg10, streakNorm, trend35, trend510,
            volatility, last3Pattern/7, parity, bigRatio10, momentum,
            hourEffect, variance5, (binary[len-2]||0)
        ];
    }

    async predict(historyNumbers, periodParity) {
        const features = this.extractFeatures(historyNumbers, periodParity);
        const prob1 = this.model1.predict(features);
        const prob2 = this.model2.predict(features);
        // For pattern matcher, need last3 binary string
        const binary = historyNumbers.map(n => n>=5?1:0);
        const last3Pattern = binary.slice(-3).join('');
        const prob3 = this.model3.predict(last3Pattern);
        
        const totalWeight = this.weights.m1 + this.weights.m2 + this.weights.m3;
        const ensembleProb = (prob1 * this.weights.m1 + prob2 * this.weights.m2 + prob3 * this.weights.m3) / totalWeight;
        
        const prediction = ensembleProb >= 0.5 ? "BIG" : "SMALL";
        const confidence = (Math.abs(ensembleProb - 0.5) * 2 * 100).toFixed(2) + "%";
        
        return { prediction, confidence, prob1, prob2, prob3, features };
    }

    update(actualBinary, features, prob1, prob2, prob3) {
        // Update each model
        this.model1.update(actualBinary, features);
        this.model2.update(actualBinary, features, 15);
        // Update pattern matcher with full history (handled separately in main loop)
        
        // Update ensemble weights based on model performance (simple online)
        this.performance.total++;
        const acc1 = 1 - Math.abs(actualBinary - prob1);
        const acc2 = 1 - Math.abs(actualBinary - prob2);
        const acc3 = 1 - Math.abs(actualBinary - prob3);
        this.performance.m1 = (this.performance.m1 * (this.performance.total-1) + acc1) / this.performance.total;
        this.performance.m2 = (this.performance.m2 * (this.performance.total-1) + acc2) / this.performance.total;
        this.performance.m3 = (this.performance.m3 * (this.performance.total-1) + acc3) / this.performance.total;
        
        // Adjust weights (softmax over recent performance)
        const exp1 = Math.exp(this.performance.m1 * 5);
        const exp2 = Math.exp(this.performance.m2 * 5);
        const exp3 = Math.exp(this.performance.m3 * 5);
        const sum = exp1+exp2+exp3;
        this.weights.m1 = exp1/sum;
        this.weights.m2 = exp2/sum;
        this.weights.m3 = exp3/sum;
    }
}

// ========== GLOBAL STATE ==========
const ensemble = new EnsemblePredictor();
let numberHistory = [];        // actual numbers (0-9) for last 50 draws
let binaryHistory = [];        // 0/1 for BIG/SMALL
let predictionsMap = new Map(); // period -> {prediction, confidence, features, probs}
let resultsHistory = [];
let totalTrades = 0;
let wins = 0;
let lastProcessedPeriod = null;
let syntheticCounter = 1000;

// ========== API FETCH ==========
async function fetchLatestResult() {
    try {
        const url = `${API_URL}?ts=${Date.now()}`;
        const res = await axios.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.ar-lottery01.com/",
                "Origin": "https://draw.ar-lottery01.com"
            },
            timeout: 15000
        });
        const list = res.data?.data?.list || res.data?.list || [];
        if (list.length > 0) {
            const item = list[0];
            const period = String(item.issue || item.issueNumber);
            const number = parseInt(item.number);
            if (period && !isNaN(number)) return { period, number };
        }
        return null;
    } catch (err) {
        console.log(`[API Error] ${err.message}`);
        return null;
    }
}

function generateSyntheticResult() {
    syntheticCounter++;
    const number = Math.floor(Math.random() * 10);
    return { period: String(syntheticCounter), number };
}

function evaluatePrediction(period, actualNumber) {
    const predObj = predictionsMap.get(period);
    if (!predObj) return false;
    
    const actualCategory = actualNumber >= 5 ? "BIG" : "SMALL";
    const predictedCategory = predObj.prediction;
    const isWin = (predictedCategory === actualCategory);
    
    totalTrades++;
    if (isWin) wins++;
    
    const actualBinary = actualCategory === "BIG" ? 1 : 0;
    // Update ensemble with actual result
    ensemble.update(actualBinary, predObj.features, predObj.prob1, predObj.prob2, predObj.prob3);
    
    // Update pattern matcher separately (needs full binary history)
    binaryHistory.push(actualBinary);
    if (binaryHistory.length > 50) binaryHistory.shift();
    ensemble.model3.update(binaryHistory);
    
    numberHistory.push(actualNumber);
    if (numberHistory.length > 50) numberHistory.shift();
    
    // Store result with emoji
    resultsHistory.unshift({
        period: period,
        sticker: isWin ? "✅ WIN" : "❌ LOSS",
        prediction: predictedCategory,
        actual: actualCategory,
        actualNumber: actualNumber,
        result: isWin ? "WIN" : "LOSS",
        confidence: predObj.confidence,
        model: "Ensemble (LR+NB+Pattern)",
        time: new Date().toLocaleTimeString()
    });
    if (resultsHistory.length > 10) resultsHistory.pop();
    
    console.log(`[RESULT] ${period} | Pred: ${predictedCategory} | Actual: ${actualCategory} (${actualNumber}) → ${isWin ? "WIN ✅" : "LOSS ❌"} | Conf: ${predObj.confidence}`);
    
    predictionsMap.delete(period);
    return isWin;
}

async function generatePrediction(currentPeriod, currentNumber) {
    const nextPeriod = String(parseInt(currentPeriod) + 1);
    const nextParity = parseInt(nextPeriod) % 2;
    // Use numberHistory for features (includes current result? No, before adding)
    const { prediction, confidence, prob1, prob2, prob3, features } = await ensemble.predict(numberHistory, nextParity);
    
    predictionsMap.set(nextPeriod, {
        prediction: prediction,
        confidence: confidence,
        features: features,
        prob1, prob2, prob3
    });
    
    console.log(`[PREDICT] Next ${nextPeriod} → ${prediction} (${confidence}) | Model scores: LR=${prob1.toFixed(3)}, NB=${prob2.toFixed(3)}, Pat=${prob3.toFixed(3)}`);
    return { period: nextPeriod, prediction, confidence };
}

async function update() {
    try {
        let current = await fetchLatestResult();
        if (!current) {
            current = generateSyntheticResult();
            console.log(`[SYNTHETIC] Period ${current.period} → ${current.number}`);
        } else {
            console.log(`[LIVE] Period ${current.period} → ${current.number}`);
        }
        
        if (lastProcessedPeriod !== current.period) {
            if (predictionsMap.has(current.period)) {
                evaluatePrediction(current.period, current.number);
            } else {
                // First run or missed period: add to history without prediction
                const actualBinary = current.number >= 5 ? 1 : 0;
                binaryHistory.push(actualBinary);
                if (binaryHistory.length > 50) binaryHistory.shift();
                numberHistory.push(current.number);
                if (numberHistory.length > 50) numberHistory.shift();
                console.log(`[INFO] Period ${current.period} added to history (no prediction)`);
            }
            lastProcessedPeriod = current.period;
        }
        
        const nextPeriod = String(parseInt(current.period) + 1);
        if (!predictionsMap.has(nextPeriod)) {
            await generatePrediction(current.period, current.number);
        }
    } catch (err) {
        console.error(`[UPDATE ERROR] ${err.message}`);
    }
}

// Start the bot
(async function start() {
    console.log(`🚀 ${NAME} starting...`);
    await update();
    setInterval(async () => {
        await update();
    }, 60000);
})();

// ========== EXPRESS ROUTES ==========
app.get('/trade', (req, res) => {
    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
    const nextPeriod = predictionsMap.keys().next().value;
    const currentPred = nextPeriod ? predictionsMap.get(nextPeriod) : { prediction: "BIG", confidence: "50.00%" };
    
    res.json({
        currentPrediction: {
            period: nextPeriod || "WAITING",
            prediction: currentPred.prediction,
            confidence: currentPred.confidence,
            model: "Ensemble (LR+NB+PatternMatcher)",
            ensembleWeights: {
                logistic: ensemble.weights.m1.toFixed(2),
                naiveBayes: ensemble.weights.m2.toFixed(2),
                pattern: ensemble.weights.m3.toFixed(2)
            },
            source: "Advanced AI 85% target",
            timestamp: new Date().toISOString()
        },
        performance: {
            totalTrades: totalTrades,
            totalWins: wins,
            totalLosses: totalTrades - wins,
            winRate: `${winRate}%`,
            targetAccuracy: "85%"
        },
        last10Predictions: resultsHistory,
        systemStatus: {
            activeModel: "Ensemble of 3 models with adaptive weighting",
            dataPoints: totalTrades,
            lastUpdate: new Date().toLocaleTimeString(),
            learningRate: "adaptive"
        }
    });
});

app.get('/', (req, res) => {
    res.json({ status: "active", name: NAME, version: "3.0 - Advanced Ensemble AI" });
});

app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
    console.log(`📡 Trade API: http://localhost:${PORT}/trade`);
});
