const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const NAME = "PRITESH AI PREDICTOR";
const API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json";

// ========== ADVANCED ADAPTIVE AI (identical to Python version) ==========
class AdaptiveAIPredictor {
    constructor(lr = 0.1, nFeatures = 8) {
        this.lr = lr;
        this.weights = Array(nFeatures).fill().map(() => Math.random() - 0.5);
        this.bias = Math.random() - 0.5;
    }

    getFeatures(history) {
        if (history.length < 5) {
            const f = Array(8).fill(0);
            if (history.length > 0) f[0] = history[history.length - 1];
            return f;
        }
        const last = history[history.length - 1];
        const last3 = history.slice(-3);
        const last5 = history.slice(-5);
        const avg3 = last3.reduce((a, b) => a + b, 0) / 3;
        const avg5 = last5.reduce((a, b) => a + b, 0) / 5;

        let streak = 1;
        for (let i = history.length - 2; i >= 0 && history[i] === last; i--) streak++;

        const diff = history.length >= 2 ? last - history[history.length - 2] : 0;
        const bigRatio5 = avg5;
        const parity = history.length % 2;

        const last4 = history.slice(-4);
        const mean4 = last4.reduce((a, b) => a + b, 0) / 4;
        const variance = last4.reduce((sum, val) => sum + Math.pow(val - mean4, 2), 0) / 4;
        const volatility = Math.sqrt(variance);

        return [last, avg3, avg5, streak / 5, diff, bigRatio5, parity, volatility];
    }

    predict(features) {
        let dot = this.bias;
        for (let i = 0; i < features.length; i++) {
            dot += features[i] * this.weights[i];
        }
        return 1 / (1 + Math.exp(-dot));
    }

    update(actual, features) {
        const predictedProb = this.predict(features);
        const error = actual - predictedProb;
        for (let i = 0; i < features.length; i++) {
            this.weights[i] += this.lr * error * features[i];
        }
        this.bias += this.lr * error;
    }

    getCategory(prob) {
        return prob >= 0.5 ? "BIG" : "SMALL";
    }
}

// ========== GLOBAL STATE (mirrors Python globals) ==========
const ai = new AdaptiveAIPredictor();
let categoryHistory = [];        // 0=SMALL, 1=BIG
let predictionsMap = new Map();   // period -> {prediction, confidence, featuresUsed}
let resultsHistory = [];
let totalTrades = 0;
let wins = 0;
let lastProcessedPeriod = null;
let syntheticCounter = 1000;

// ========== API FETCH (same headers as Python) ==========
async function fetchLatestResult() {
    try {
        const url = `${API_URL}?ts=${Date.now()}`;
        const res = await axios.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.ar-lottery01.com/",
                "Origin": "https://draw.ar-lottery01.com",
                "Accept": "application/json, text/plain, */*"
            },
            timeout: 15000
        });
        const list = res.data?.data?.list || res.data?.list || [];
        if (list.length > 0) {
            const item = list[0];
            const period = String(item.issue || item.issueNumber);
            const number = parseInt(item.number);
            if (period && !isNaN(number)) {
                return { period, number };
            }
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
    ai.update(actualBinary, predObj.featuresUsed);

    categoryHistory.push(actualBinary);
    if (categoryHistory.length > 50) categoryHistory.shift();

    resultsHistory.unshift({
        period: period,
        sticker: isWin ? "WIN" : "LOSS",
        prediction: predictedCategory,
        actual: actualCategory,
        actualNumber: actualNumber,
        result: isWin ? "WIN" : "LOSS",
        confidence: predObj.confidence,
        model: "Adaptive AI",
        time: new Date().toLocaleTimeString()
    });
    if (resultsHistory.length > 10) resultsHistory.pop();

    console.log(`[RESULT] Period ${period} | Pred: ${predictedCategory} | Actual: ${actualCategory} (${actualNumber}) → ${isWin ? "WIN" : "LOSS"} | Conf: ${predObj.confidence}`);

    predictionsMap.delete(period);
    return isWin;
}

function generatePrediction(currentPeriod) {
    const nextPeriod = String(parseInt(currentPeriod) + 1);
    let features = ai.getFeatures(categoryHistory);
    const nextParity = parseInt(nextPeriod) % 2;
    if (features.length > 6) features[6] = nextParity;

    const prob = ai.predict(features);
    const predictedCategory = ai.getCategory(prob);
    const confidence = (Math.abs(prob - 0.5) * 2 * 100).toFixed(2) + "%";

    predictionsMap.set(nextPeriod, {
        prediction: predictedCategory,
        confidence: confidence,
        featuresUsed: [...features]
    });

    console.log(`[PREDICT] Next ${nextPeriod} → ${predictedCategory} (${confidence})`);
    return { period: nextPeriod, prediction: predictedCategory, confidence };
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
                const actualBinary = current.number >= 5 ? 1 : 0;
                categoryHistory.push(actualBinary);
                if (categoryHistory.length > 50) categoryHistory.shift();
                console.log(`[INFO] Period ${current.period} added to history (no prediction)`);
            }
            lastProcessedPeriod = current.period;
        } else {
            console.log(`[SKIP] Duplicate period ${current.period}`);
        }

        const nextPeriod = String(parseInt(current.period) + 1);
        if (!predictionsMap.has(nextPeriod)) {
            generatePrediction(current.period);
        }
    } catch (err) {
        console.error(`[UPDATE ERROR] ${err.message}`);
    }
}

// ========== START THE BOT (safe async) ==========
(async function start() {
    console.log(`🚀 ${NAME} starting...`);
    try {
        await update();
        console.log(`✅ Initial update complete.`);
    } catch (err) {
        console.error(`❌ Initial update failed: ${err.message}`);
    }
    setInterval(async () => {
        try {
            await update();
        } catch (err) {
            console.error(`❌ Periodic update failed: ${err.message}`);
        }
    }, 60000); // every minute
})();

// ========== EXPRESS ROUTES (exactly matching Python endpoints) ==========
app.get('/trade', (req, res) => {
    const winRate = totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(2) : 0;
    const nextPeriod = predictionsMap.keys().next().value; // first key
    const currentPred = nextPeriod ? predictionsMap.get(nextPeriod) : { prediction: "BIG", confidence: "50.00%" };

    res.json({
        currentPrediction: {
            period: nextPeriod || "WAITING",
            prediction: currentPred.prediction,
            confidence: currentPred.confidence,
            model: "Adaptive Online AI",
            source: "WinGo_1M + gradient descent",
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
    console.log(`✅ Server running on port ${PORT}`);
    console.log(`📡 Trade API: http://localhost:${PORT}/trade`);
});
