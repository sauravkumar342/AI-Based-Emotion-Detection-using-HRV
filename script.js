const API = "http://127.0.0.1:8000";

let chart;

function generateECG() {
    return Array.from({ length: 200 }, () => Math.random() * 2);
}

function drawChart(data) {

    const ctx = document.getElementById('chart');

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                label: 'ECG Signal',
                data: data
            }]
        }
    });
}

function predict() {

    let ecg = generateECG();

    drawChart(ecg);

    fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ecg)
    })
        .then(res => res.json())
        .then(d => {

            document.getElementById("prediction").innerText = d.prediction;
            document.getElementById("hr").innerText = d.heart_rate + " bpm";
            document.getElementById("stress").innerText = d.stress;
            document.getElementById("risk").innerText = d.risk;

            if (d.risk === "HIGH") {
                alert("🚨 HIGH RISK DETECTED");
            }
        });
}

function runXAI() {

    let ecg = generateECG();

    fetch(API + "/xai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(ecg)
    })
        .then(res => res.json())
        .then(d => {
            document.getElementById("xai").src =
                API + "/" + d.image + "?t=" + Date.now();
        });
}



document.getElementById("liveData").innerText =
    `HR: ${data.heart_rate}
 Stress: ${data.stress}
 Emotion: ${data.emotion}
 Risk: ${data.risk}`;