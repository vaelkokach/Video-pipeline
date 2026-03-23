const statusEl = document.getElementById("status");
const studentsBody = document.getElementById("students-body");
const alertsEl = document.getElementById("alerts");

function levelClass(level) {
  return level || "neutral";
}

function renderStudents(students) {
  studentsBody.innerHTML = "";
  for (const student of students) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${student.student_id}</td>
      <td><span class="tag ${levelClass(student.attention_level)}">${student.attention_level}</span></td>
      <td>${(student.loss_probability * 100).toFixed(1)}%</td>
      <td>${student.emotion || "-"}</td>
      <td>${student.action || "-"}</td>
    `;
    studentsBody.appendChild(tr);
  }
}

function appendAlerts(events) {
  for (const evt of events) {
    const li = document.createElement("li");
    li.textContent = `Student ${evt.student_id} - ${evt.event_type} (${evt.level})`;
    alertsEl.prepend(li);
  }
  while (alertsEl.children.length > 20) {
    alertsEl.removeChild(alertsEl.lastChild);
  }
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/realtime`);

  ws.onopen = () => {
    statusEl.textContent = "Realtime stream connected.";
  };
  ws.onclose = () => {
    statusEl.textContent = "Realtime stream disconnected. Reconnecting...";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => {
    statusEl.textContent = "Realtime stream error.";
  };
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    renderStudents(payload.students || []);
    appendAlerts(payload.events || []);
  };
}

connect();
