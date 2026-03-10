// Configuration — set this to your FastAPI server URL
var API_BASE_URL = 'http://localhost:8000';

var client = ZAFClient.init();

document.getElementById('generate-btn').addEventListener('click', function() {
  var btn = document.getElementById('generate-btn');
  var loading = document.getElementById('loading');
  var result = document.getElementById('result');
  var error = document.getElementById('error');

  btn.disabled = true;
  loading.style.display = 'block';
  result.style.display = 'none';
  error.style.display = 'none';

  client.get('ticket.id').then(function(data) {
    var ticketId = data['ticket.id'];

    fetch(API_BASE_URL + '/zendesk/generate-draft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticket_id: ticketId })
    })
    .then(function(response) {
      if (!response.ok) {
        throw new Error('Server error: ' + response.status);
      }
      return response.json();
    })
    .then(function(data) {
      document.getElementById('draft').textContent = data.draft;

      var citationsEl = document.getElementById('citations');
      citationsEl.innerHTML = '<strong>Sources:</strong><br>' + (data.citations || []).join('<br>');

      var escEl = document.getElementById('escalation');
      if (data.escalation && data.escalation.needs_escalation) {
        escEl.className = 'escalate';
        escEl.textContent = 'ESCALATE: ' + data.escalation.reason;
      } else {
        escEl.className = 'no-escalate';
        escEl.textContent = 'No escalation needed';
      }

      document.getElementById('meta').textContent =
        'Language: ' + (data.detected_language || '').toUpperCase() +
        ' | Complexity: ' + (data.complexity || '');

      loading.style.display = 'none';
      result.style.display = 'block';
      btn.disabled = false;
    })
    .catch(function(err) {
      loading.style.display = 'none';
      error.style.display = 'block';
      error.textContent = 'Error: ' + err.message;
      btn.disabled = false;
    });
  });
});
