const $ = (id) => document.getElementById(id);

const paletteClass = (key) => {
  if (key === "Home Team") return "home";
  if (key === "Draw") return "draw";
  if (key === "Away Team") return "away";
  return "draw";
};

async function loadMeta() {
  const status = $("status");
  const res = await fetch("/api/meta");
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    status.textContent =
      err.detail ||
      "Model not trained. From the project folder run: python scripts/train_model.py";
    status.className = "status err";
    $("submit-btn").disabled = true;
    return;
  }
  const data = await res.json();
  const ht = $("home_team");
  const at = $("away_team");
  ht.innerHTML = "";
  at.innerHTML = "";
  for (const t of data.teams) {
    ht.add(new Option(t, t));
    at.add(new Option(t, t));
  }
  if (data.teams.length >= 2) {
    ht.selectedIndex = 0;
    at.selectedIndex = 1;
  }
  status.textContent = "Ready — model loaded.";
  status.className = "status ok";
}

function defaultDate() {
  const d = new Date();
  $("date").value = d.toISOString().slice(0, 10);
  $("year").value = d.getFullYear();
}

$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const body = {
    model: fd.get("model"),
    home_team: fd.get("home_team"),
    away_team: fd.get("away_team"),
    year: Number(fd.get("year")),
    date: fd.get("date"),
    possession_home: Number(fd.get("possession_home")),
    possession_away: Number(fd.get("possession_away")),
    shots_home: Number(fd.get("shots_home")),
    shots_away: Number(fd.get("shots_away")),
    corners_home: Number(fd.get("corners_home")),
    corners_away: Number(fd.get("corners_away")),
    fouls_home: Number(fd.get("fouls_home")),
    fouls_away: Number(fd.get("fouls_away")),
  };

  const btn = $("submit-btn");
  btn.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const out = await res.json();
    if (!res.ok) {
      const msg = Array.isArray(out.detail)
        ? out.detail.map((d) => d.msg || d).join(" ")
        : out.detail || res.statusText;
      alert(msg);
      return;
    }
    $("result").classList.remove("hidden");
    $("prediction-text").textContent = out.prediction_display;

    const bars = $("prob-bars");
    bars.innerHTML = "";
    const order = Array.isArray(out.class_order) ? out.class_order : ["Home Team", "Draw", "Away Team"];
    for (const k of order) {
      const p = out.probabilities[k] ?? 0;
      const row = document.createElement("div");
      row.className = "bar-row";
      const label =
        k === "Home Team" ? "Home" : k === "Away Team" ? "Away" : "Draw";
      row.innerHTML = `
        <span>${label}</span>
        <div class="bar-track"><div class="bar-fill ${paletteClass(
          k
        )}" style="width:${(p * 100).toFixed(1)}%"></div></div>
        <span>${(p * 100).toFixed(1)}%</span>
      `;
      bars.appendChild(row);
    }
  } finally {
    btn.disabled = false;
  }
});

defaultDate();
loadMeta().catch(() => {
  $("status").textContent = "Cannot reach API. Start server: uvicorn app.main:app --reload";
  $("status").className = "status err";
});
