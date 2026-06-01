'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const logger = require('./logger');

const PYTHON_API_URL = (process.env.PYTHON_API_URL || 'http://localhost:8000').replace(/\/$/, '');
const WEBHOOK_SECRET = process.env.WEBHOOK_SECRET || 'change-me-in-production';
const WEBHOOK_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes — LLM inference can be slow
const MAX_RETRIES = 3;
const RETRY_BASE_MS = 1500;

// Shared mutable state exposed to the Express server for status reporting.
const state = {
  connected: false,
  pendingQR: null,
};

// ── Webhook delivery ───────────────────────────────────────────────────────────

async function deliverWebhook(payload, attempt = 1) {
  try {
    await axios.post(`${PYTHON_API_URL}/webhook/whatsapp`, payload, {
      headers: {
        Authorization: `Bearer ${WEBHOOK_SECRET}`,
        'Content-Type': 'application/json',
      },
      timeout: WEBHOOK_TIMEOUT_MS,
    });
    logger.debug({ from: payload.from, attempt }, 'Webhook delivered');
  } catch (err) {
    const status = err.response?.status;
    logger.warn({ from: payload.from, attempt, status, msg: err.message }, 'Webhook delivery failed');

    // Do not retry on 4xx (permanent errors such as 403 Forbidden).
    if (status && status >= 400 && status < 500) return;

    if (attempt < MAX_RETRIES) {
      const delay = RETRY_BASE_MS * attempt;
      logger.info({ from: payload.from, delay, nextAttempt: attempt + 1 }, 'Retrying webhook...');
      await new Promise((r) => setTimeout(r, delay));
      return deliverWebhook(payload, attempt + 1);
    }

    logger.error({ from: payload.from }, 'Webhook delivery permanently failed after max retries');
  }
}

// ── Client factory ────────────────────────────────────────────────────────────

function createClient() {
  const client = new Client({
    authStrategy: new LocalAuth({
      dataPath: process.env.AUTH_DATA_PATH || '/app/.wwebjs_auth',
    }),
    puppeteer: {
      executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--no-first-run',
        '--no-zygote',
        '--single-process',
        '--disable-extensions',
      ],
    },
  });

  client.on('qr', (qr) => {
    state.pendingQR = qr;
    logger.info('QR code ready — scan with WhatsApp on your phone to authenticate');
    qrcode.generate(qr, { small: true });
  });

  client.on('authenticated', () => {
    state.pendingQR = null;
    logger.info('WhatsApp session authenticated');
  });

  client.on('auth_failure', (msg) => {
    logger.error({ msg }, 'WhatsApp authentication failed — delete auth volume and restart');
  });

  client.on('ready', () => {
    state.connected = true;
    state.pendingQR = null;
    logger.info('WhatsApp client is ready');
  });

  client.on('disconnected', (reason) => {
    state.connected = false;
    logger.warn({ reason }, 'WhatsApp client disconnected');
  });

  client.on('message', async (message) => {
    // Ignore: broadcast channels, self-sent messages, group chats.
    if (message.from === 'status@broadcast') return;
    if (message.fromMe) return;
    if (message.from.endsWith('@g.us')) {
      logger.debug({ from: message.from }, 'Ignoring group message');
      return;
    }

    logger.info({ from: message.from, type: message.type }, 'Message received');

    await deliverWebhook({
      from: message.from,
      body: message.body || '',
      timestamp: Math.floor(message.timestamp),
      type: message.type,
    });
  });

  return client;
}

module.exports = { createClient, state };
