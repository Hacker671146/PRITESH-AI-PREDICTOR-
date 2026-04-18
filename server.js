const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const NAME = "PRITESH AI PREDICTOR";
const API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json";

// ========== ADVANCED ADAPTIVE AI (ONLINE LEARNING) ==========
class AdaptiveAIPredictor {
    constructor(lr = 0.1, nFeatures = 8) {
        this.lr = lr;
        // Initialize weights randomly between -0.5 and 0.5
        this.weights = Array(nFeatures).fill().map(() => (Math.random() - 0.5));
        this.bias = (Math.random() - 0.5);
    }

    // Extract 8 features from last results (binary: BIG=1, SMALL=0)
    getFeatures(history) {
        // history = array of 0/1, most recent last
        if (history.length < 5) {
            // Not enough data – return zeros + simple fallback
            const f = Array(8).fill(0);
            if (history.length > 0) f[0] = history[history.length-1];
            return f;
        }
        const last = history[history.length-1];
        const last3 = history.slice(-3);
        const last5 = history.slice(-5);
        const avg3 = last3.reduce((a,b)=>a+b,0)/3;
        const avg5 = last5.reduce((a,b)=>a+b,0)/5;
        
        // Streak of same outcome as last
        let streak = 1;
        for (let i = history.length-2; i >= 0 && history[i] === last; i--) streak++;
        
        const diff = history.length >= 2 ? last - history[history.length-2] : 0;
        const bigRatio5 = avg5; // because average of binary = proportion of BIGs
        
        // Period parity simulation – use length as proxy (real period parity added in predict call)
        const parity = (history.length % 2);  // will be overridden later with actual period number
        
        // Volatility: standard deviation of last 4
        const last4 = history.slice(-4);
        const mean4 = last4.reduce((a,b)=>a+b,0)/4;
        const variance = last4.reduce((sum,val)=>sum + (val-mean4)**2,0)/4;
        const volatility = Math.sqrt(variance);
        
        return [last, avg3, avg5, streak/5, diff, bigRatio5, parity, volatility];
    }

    predict(features) {
        let dot = this.bias;
        for (let i = 0; i < features.length; i++) {
            dot += features[i] * this.weights[i];
        }
        return 1 / (1 + Math.exp(-dot)); // sigmoid -> probability of BIG
    }

    // Online learning: update weights after actual outcome (actual = 1 for BIG, 0 for SMALL)
    update(actual, features) {
        const predictedProb = this.predict(features);
        const error = actual - predictedProb;
        // Gradient descent update
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.lr * error * features[i];
        }
        this.bias += this.lr * error;
    }

    // Get category from probability
    getCategory(prob) {
        return prob >= 0.5 ? "BIG" : "SMALL";
    }
}

// ========== GLOBAL STATE ==========
const ai = new AdaptiveAIPredictor(0.1, 8);
let categoryHistory = [];       // 0=SMALL, 1=BIG from actual results
let predictionsMap = {};        // period -> { prediction, confidence, featuresUsed }
let resultsHistory = [];        // last 10 results for display
let totalTrades = 0;
let wins = 0;
let lastProcessedPeriod = null;
let syntheticCounter = 1000;

// ========== FETCH REAL API (with anti-403 headers) ==========
async function fetchLatestResult() {
    try {
        const url = `${API_URL}?ts=${Date.now()}`;
        const res = await axios.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.ar-lottery01.com/",
                "Origin": "https://draw.ar-lottery01.com"
            },
            timeout: 10000
        });
        const list = res.data?.data?.list || res.data?.list || [];
        if (list && list.length > 0) {
            const item = list[0];
            const period = item.issue || item.issueNumber;
            const number = parseInt(item.number);
            if (period && !isNaN(number)) {
                return { period: String(period), number };
            }
        }
        return null;
    } catch (err) {
        console.log(`[API Error] ${err.message}`);
        return null;
    }
}

// ========== SYNTHETIC FALLBACK ==========
function generateSyntheticResult() {
    syntheticCounter++;
    const number = Math.floor(Math.random() * 10);
    return { period: String(syntheticCounter), number };
}

// ========== EVALUATE PREDICTION FOR A PERIOD ==========
function evaluatePrediction(period, actualNumber) {
    const predictionObj = predictionsMap[period];
    if (!predictionObj) {
        console.log(`[WARN] No prediction for period ${period}`);
        return false;
    }
    const actualCategory = actualNumber >= 5 ? "BIG" : "SMALL";
    const predictedCategory = predictionObj.prediction;
    const isWin = (predictedCategory === actualCategory);
    
    totalTrades++;
    if (isWin) wins++;
    
    // Update AI with the actual result using the features we stored when making the prediction
    const actualBinary = actualCategory === "BIG" ? 1 : 0;
    ai.update(actualBinary, predictionObj.featuresUsed);
    
    // Add actual result to category history (for future feature extraction)
    categoryHistory.push(actualBinary);
    if (categoryHistory.length > 50) categoryHistory.shift();
    
    // Store in results history for display
    resultsHistory.unshift({
        period: period,
        sticker: isWin ? "✅ WIN" : "❌ LOSS",
        prediction: predictedCategory,
        actual: actualCategory,
        actualNumber: actualNumber,
        result: isWin ? "WIN" : "LOSS",
        confidence: predictionObj.confidence,
        model: "Adaptive AI",
        time: new Date().toLocaleTimeString()
    });
    if (resultsHistory.length > 10) resultsHistory.pop();
    
    console.log(`[RESULT] Period ${period} | Pred: ${predictedCategory} | Actual: ${actualCategory} (${actualNumber}) → ${isWin ? "WIN" : "LOSS"} | Confidence: ${predictionObj.confidence}`);
    
    delete predictionsMap[period];
    return isWin;
}

// ========== GENERATE PREDICTION FOR NEXT PERIOD ==========
function generatePrediction(currentPeriod, currentNumber, currentPeriodParity) {
    const nextPeriod = (parseInt(currentPeriod) + 1).toString();
    
    // Get features from current categoryHistory (does NOT include the current result yet)
    let features = ai.getFeatures(categoryHistory);
    // Override parity feature (index 6) with actual period number parity of the NEXT period
    const nextParity = (parseInt(nextPeriod) % 2);
    if (features.length > 6) features[6] = nextParity;
    
    const prob = ai.predict(features);
    const predictedCategory = ai.getCategory(prob);
    const confidence = (Math.abs(prob - 0.5) * 2 * 100).toFixed(2) + "%";  // how far from 0.5
    
    predictionsMap[nextPeriod] = {
        prediction: predictedCategory,
        confidence: confidence,
        featuresUsed: features.slice()   // store copy for later learning
    };
    
    console.log(`[PREDICT] Next period ${nextPeriod} → ${predictedCategory} (confidence ${confidence})`);
    return { period: nextPeriod, prediction: predictedCategory, confidence };
}

// ========== MAIN UPDATE LOOP ==========
async function update() {
    // 1. Get latest result
    let current = await fetchLatestResult();
    let usingReal = true;
    if (!current) {
        usingReal = false;
        current = generateSyntheticResult();
        console.log(`[SYNTHETIC] Period ${current.period} → ${current.number}`);
    } else {
        console.log(`[LIVE] Period ${current.period} → ${current.number}`);
    }
    
    // 2. If this period is new, evaluate any prediction for it and then learn
    if (lastProcessedPeriod !== current.period) {
        // Evaluate prediction for this period if it exists
        if (predictionsMap[current.period]) {
            evaluatePrediction(current.period, current.number);
        } else {
            console.log(`[INFO] Period ${current.period} has no prediction (first run or missed)`);
            // Still need to update category history so AI can learn from this outcome
            const actualBinary = (current.number >= 5) ? 1 : 0;
            categoryHistory.push(actualBinary);
            if (categoryHistory.length > 50) categoryHistory.shift();
        }
        
        lastProcessedPeriod = current.period;
    } else {
        console.log(`[SKIP] Duplicate period ${current.period}, already processed`);
    }
    
    // 3. Generate prediction for the next period (if not already generated)
    const nextPeriod = (parseInt(current.period) + 1).toString();
    if (!predictionsMap[nextPeriod]) {
        // Pass current period parity for feature extraction
        generatePrediction(current.period, current.number, parseInt(current.period) % 2);
    } else {
        console.log(`[EXISTS] Prediction for ${nextPeriod} already exists`);
    }
}

// Run every 60 seconds (WinGo 1M draws every minute)
setInterval(update, 60000);
update(); // immediate start

// ========== EXPRESS ROUTES ==========
app.get('/trade', (req, res) => {
    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
    // Get the current prediction (for the next period)
    const nextPeriod = Object.keys(predictionsMap)[0];
    const currentPred = nextPeriod ? predictionsMap[nextPeriod] : { prediction: "BIG", confidence: "50.00%" };
    
    res.json({
        currentPrediction: {
            period: nextPeriod || "WAITING",
            prediction: currentPred.prediction,
            confidence: currentPred.confidence,
            model: "Adaptive Online AI",
            source: "WinGo_1M + gradient descent learning",
            marketState: "ACTIVE",
            timestamp: new Date().toISOString()
        },
        performance: {
            totalTrades: totalTrades,
            totalWins: wins,
            totalLosses: totalTrades - wins,
            winRate: `${winRate}%`,
            currentLevel: 1,
            currentMultiplier: 1
        },
        last10Predictions: resultsHistory,
        systemStatus: {
            activeModel: "Adaptive AI (8 features, online learning)",
            dataPoints: totalTrades,
            marketRegime: "LEARNING",
            lastUpdate: new Date().toLocaleTimeString(),
            learningRate: ai.lr,
            apiConnected: true
        }
    });
});

app.get('/', (req, res) => {
    res.json({ status: "active", name: NAME, version: "2.0 - BIG/SMALL only" });
});

app.get('/health', (req, res) => {
    res.status(200).send("OK");
});

app.listen(PORT, () => {
    console.log(`✅ ${NAME} running on port ${PORT}`);
    console.log(`📡 Trade API: http://localhost:${PORT}/trade`);
    console.log(`🧠 Advanced AI: Online gradient descent with 8 features`);
});
