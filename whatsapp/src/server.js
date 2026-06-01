'use strict';

const express = require('express');
const { createClient, state } = require('./client');
const logger = require('./logger');

const PORT = parseInt(process.env.PORT || '3001', 10);
const app = express();
app.use(express.json());

// ── WhatsApp client ───────────────────────────────────────────────────────────

const client = createClient();

// ── REST API ──────────────────────────────────────────────────────────────────

/** POST /send — called by the Python API to send a message to the user. */
app.post('/send', async (req, res) => {
  const { to, message } = req.body;
  if (!to || typeof message !== 'string') {
    return res.status(400).json({ error: 'Missing required fields: to, message' });
  }
  if (!state.connected) {
    return res.status(503).json({ error: 'WhatsApp client not connected' });
  }
  try {
    const result = await client.sendMessage(to, message);
    logger.info({ to, length: message.length }, 'Message sent');
    res.json({ success: true, messageId: result.id._serialized });
  } catch (err) {
    logger.error({ to, err: err.message }, 'Failed to send message');
    res.status(500).json({ error: 'Failed to send message', details: err.message });
  }
});

/** POST /typing — signal "typing..." to the user. Best-effort. */
app.post('/typing', async (req, res) => {
  const { to } = req.body;
  if (!to) return res.status(400).json({ error: 'Missing field: to' });
  try {
    const chat = await client.getChatById(to);
    await chat.sendStateTyping();
    res.json({ success: true });
  } catch (err) {
    // Typing is best-effort — don't alarm callers.
    res.json({ success: false, reason: err.message });
  }
});

/** GET /status — WhatsApp connection state for readiness checks. */
app.get('/status', (req, res) => {
  res.json({
    state: state.connected ? 'CONNECTED' : 'INITIALIZING',
    qr_pending: state.pendingQR !== null,
  });
});

/** GET /health — liveness probe used by Docker and the Python API readiness check. */
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// ── Boot ──────────────────────────────────────────────────────────────────────

const server = app.listen(PORT, () => {
  logger.info({ port: PORT }, 'WhatsApp bridge HTTP server listening');
});

client.initialize().catch((err) => {
  logger.error({ err: err.message }, 'WhatsApp client failed to initialize');
  process.exit(1);
});

// ── Graceful shutdown ─────────────────────────────────────────────────────────

async function shutdown(signal) {
  logger.info({ signal }, 'Shutdown signal received');
  server.close();
  try {
    await client.destroy();
  } catch (_) {
    // Ignore destroy errors during shutdown
  }
  process.exit(0);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
