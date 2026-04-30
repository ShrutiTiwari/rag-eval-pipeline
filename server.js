const express = require('express');
const path = require('path');
const cors = require('cors');
require('dotenv').config();

const documentViewerRoutes = require('./src/routes/documentViewerRoutes');

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/api/rag-docs', documentViewerRoutes);

// Serve index.html for root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`🚀 rag-abrsm-exam running at http://localhost:${port}`);
  console.log(`📄 Dashboard: http://localhost:${port}/`);
  console.log(`💬 Chat:      http://localhost:${port}/api/rag-docs/chat/test`);
  console.log(`🔍 Search:    http://localhost:${port}/api/rag-docs/vector-search/test`);
  console.log(`📊 Evaluate:  http://localhost:${port}/api/rag-docs/evaluate`);
});
