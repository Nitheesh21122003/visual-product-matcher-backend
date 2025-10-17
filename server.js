const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

// Load product database (json file)
const products = JSON.parse(fs.readFileSync('products.json'));

// Endpoint to get all products
app.get('/api/products', (_, res) => {
  res.json(products);
});

// Endpoint to upload an image and get matches
app.post('/api/match', upload.single('image'), (req, res) => {
  const imagePath = req.file.path;

  // Spawn Python matcher
  const py = spawn('python', ['matcher.py', imagePath]);

  let results = '';
  let errors = '';

  py.stdout.on('data', (data) => {
    results += data.toString();
  });

  py.stderr.on('data', (data) => {
    errors += data.toString();
  });

  py.on('close', (code) => {
    console.log("Python stdout:", results);
    console.error("Python stderr:", errors);

    // Delete the uploaded file
    fs.unlink(imagePath, (err) => {
      if (err) console.error("Error deleting file: ", err);
    });

    if (code !== 0) {
      res.status(500).json({ error: `Matcher exited with code ${code}`, details: errors });
      return;
    }

    try {
      const jsonResult = JSON.parse(results);
      res.json(jsonResult);
    } catch (err) {
      res.status(500).json({ error: "Invalid JSON output from matcher", details: results });
    }
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
