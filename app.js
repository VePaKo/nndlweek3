// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = parseCSVLine(lines[0]);
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index];
            });
            data.push(row);
        } else {
            console.warn(`Skipping line ${i+1}: expected ${headers.length} columns, got ${values.length}`);
        }
    }
    return data;
}

// Helper function to parse a single CSV line, handling quoted fields
function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    let quoteChar = '"';
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === quoteChar) {
            if (inQuotes && nextChar === quoteChar) {
                // Escaped quote inside quotes
                current += quoteChar;
                i++; // Skip next quote
            } else {
                // Toggle quote state
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            values.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    
    // Add the last field
    values.push(current);
    return values;
}

// Alternative: Using a more robust CSV parser (recommended)
function parseCSVRobust(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const row = {};
        let currentIndex = 0;
        let currentField = '';
        let inQuotes = false;
        
        for (let j = 0; j < lines[i].length; j++) {
            const char = lines[i][j];
            
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                row[headers[currentIndex]] = currentField.trim();
                currentField = '';
                currentIndex++;
            } else {
                currentField += char;
            }
        }
        
        // Don't forget the last field
        row[headers[currentIndex]] = currentField.trim();
        
        if (Object.keys(row).length === headers.length) {
            data.push(row);
        }
    }
    return data;
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}


// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Convert data types first
    convertDataTypes();
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    
    // Fixed survival count - handle both string and number types
    const survivalCount = trainData.filter(row => {
        const survived = row[TARGET_FEATURE];
        return survived === 1 || survived === '1' || survived === true;
    }).length;
    
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature - FIXED
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => {
            const value = row[feature];
            return value === null || value === undefined || value === '' || isNaN(value);
        }).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}% (${missingCount} missing)</li>`;
    });
    missingInfo += '</ul>';
    
    // Add numerical statistics
    let numericalStats = '<h4>Numerical Features Statistics:</h4><ul>';
    NUMERICAL_FEATURES.forEach(feature => {
        const values = trainData.map(row => parseFloat(row[feature])).filter(val => !isNaN(val));
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const min = Math.min(...values);
            const max = Math.max(...values);
            numericalStats += `<li>${feature}: mean=${mean.toFixed(2)}, min=${min}, max=${max}, count=${values.length}</li>`;
        }
    });
    numericalStats += '</ul>';
    
    // Add categorical value counts
    let categoricalStats = '<h4>Categorical Features Value Counts:</h4><ul>';
    CATEGORICAL_FEATURES.forEach(feature => {
        const valueCounts = {};
        trainData.forEach(row => {
            const value = row[feature];
            if (value !== null && value !== undefined && value !== '') {
                valueCounts[value] = (valueCounts[value] || 0) + 1;
            }
        });
        
        categoricalStats += `<li>${feature}:`;
        categoricalStats += '<ul>';
        Object.entries(valueCounts).forEach(([value, count]) => {
            const percent = (count / trainData.length * 100).toFixed(1);
            categoricalStats += `<li>"${value}": ${count} (${percent}%)</li>`;
        });
        categoricalStats += '</ul></li>';
    });
    categoricalStats += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}${numericalStats}${categoricalStats}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// New function to convert data types
function convertDataTypes() {
    // Convert training data
    trainData.forEach(row => {
        // Convert numerical features
        NUMERICAL_FEATURES.forEach(feature => {
            if (row[feature] !== null && row[feature] !== undefined && row[feature] !== '') {
                const numValue = parseFloat(row[feature]);
                row[feature] = isNaN(numValue) ? null : numValue;
            } else {
                row[feature] = null;
            }
        });
        
        // Convert target feature to number
        if (row[TARGET_FEATURE] !== null && row[TARGET_FEATURE] !== undefined) {
            row[TARGET_FEATURE] = parseInt(row[TARGET_FEATURE]) || 0;
        }
        
        // Convert categorical features - ensure they are strings
        CATEGORICAL_FEATURES.forEach(feature => {
            if (row[feature] !== null && row[feature] !== undefined) {
                row[feature] = String(row[feature]).trim();
            } else {
                row[feature] = '';
            }
        });
    });
    
    // Convert test data similarly (but without target feature)
    if (testData) {
        testData.forEach(row => {
            NUMERICAL_FEATURES.forEach(feature => {
                if (row[feature] !== null && row[feature] !== undefined && row[feature] !== '') {
                    const numValue = parseFloat(row[feature]);
                    row[feature] = isNaN(numValue) ? null : numValue;
                } else {
                    row[feature] = null;
                }
            });
            
            CATEGORICAL_FEATURES.forEach(feature => {
                if (row[feature] !== null && row[feature] !== undefined) {
                    row[feature] = String(row[feature]).trim();
                } else {
                    row[feature] = '';
                }
            });
        });
    }
}

// Fixed createVisualizations function
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Ensure data types are converted
    convertDataTypes();
    
    // Survival by Sex - FIXED: handle different data types
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined && row.Survived !== null) {
            const sex = String(row.Sex).trim();
            const survived = parseInt(row.Survived) || 0;
            
            if (!survivalBySex[sex]) {
                survivalBySex[sex] = { survived: 0, total: 0 };
            }
            survivalBySex[sex].total++;
            if (survived === 1) {
                survivalBySex[sex].survived++;
            }
        }
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    if (sexData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Sex', tab: 'Charts' },
            sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
            { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
        );
    }
    
    // Survival by Pclass - FIXED: handle different data types
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined && row.Survived !== null) {
            const pclass = String(row.Pclass).trim();
            const survived = parseInt(row.Survived) || 0;
            
            if (!survivalByPclass[pclass]) {
                survivalByPclass[pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[pclass].total++;
            if (survived === 1) {
                survivalByPclass[pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    if (pclassData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
            pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
            { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
        );
    }
    
    // Age distribution by survival - additional useful chart
    const survivedAges = trainData
        .filter(row => row.Survived === 1 && row.Age !== null && !isNaN(row.Age))
        .map(row => row.Age);
    
    const notSurvivedAges = trainData
        .filter(row => row.Survived === 0 && row.Age !== null && !isNaN(row.Age))
        .map(row => row.Age);
    
    if (survivedAges.length > 0 || notSurvivedAges.length > 0) {
        const ageData = [
            { values: survivedAges.map(age => ({ x: age })), series: 'Survived' },
            { values: notSurvivedAges.map(age => ({ x: age })), series: 'Not Survived' }
        ];
        
        tfvis.render.histogram(
            { name: 'Age Distribution by Survival', tab: 'Charts' },
            ageData,
            { xLabel: 'Age', yLabel: 'Count' }
        );
    }
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Also fix the preview table to handle different data types better
function createPreviewTable(data) {
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            // Handle different types of missing values and display appropriately
            if (value === null || value === undefined || value === '') {
                td.textContent = 'NULL';
                td.style.color = '#999';
                td.style.fontStyle = 'italic';
            } else if (typeof value === 'number' && isNaN(value)) {
                td.textContent = 'NaN';
                td.style.color = '#999';
                td.style.fontStyle = 'italic';
            } else {
                td.textContent = value;
            }
            td.style.border = '1px solid #ddd';
            td.style.padding = '8px';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Ensure data types are converted
    convertDataTypes();
    
    // Survival by Sex - FIXED DATA FORMAT
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined && row.Survived !== null) {
            const sex = String(row.Sex).trim();
            const survived = parseInt(row.Survived) || 0;
            
            if (!survivalBySex[sex]) {
                survivalBySex[sex] = { survived: 0, total: 0 };
            }
            survivalBySex[sex].total++;
            if (survived === 1) {
                survivalBySex[sex].survived++;
            }
        }
    });
    
    // Convert to correct format for tfjs-vis
    const sexChartData = Object.entries(survivalBySex).map(([sex, stats]) => {
        const survivalRate = (stats.survived / stats.total) * 100;
        return {
            index: sex,
            value: survivalRate
        };
    });
    
    if (sexChartData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Sex', tab: 'Charts' },
            sexChartData,
            { 
                xLabel: 'Sex', 
                yLabel: 'Survival Rate (%)',
                height: 300
            }
        );
    }
    
    // Survival by Pclass - FIXED DATA FORMAT
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Pclass !== null && row.Survived !== undefined && row.Survived !== null) {
            const pclass = String(row.Pclass).trim();
            const survived = parseInt(row.Survived) || 0;
            
            if (!survivalByPclass[pclass]) {
                survivalByPclass[pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[pclass].total++;
            if (survived === 1) {
                survivalByPclass[pclass].survived++;
            }
        }
    });
    
    const pclassChartData = Object.entries(survivalByPclass).map(([pclass, stats]) => {
        const survivalRate = (stats.survived / stats.total) * 100;
        return {
            index: `Class ${pclass}`,
            value: survivalRate
        };
    });
    
    if (pclassChartData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
            pclassChartData,
            { 
                xLabel: 'Passenger Class', 
                yLabel: 'Survival Rate (%)',
                height: 300
            }
        );
    }
    
    // Alternative: Use a different approach for survival by passengers
    // This creates a pie chart showing overall survival distribution
    const overallSurvival = {
        survived: trainData.filter(row => row.Survived === 1).length,
        notSurvived: trainData.filter(row => row.Survived === 0).length
    };
    
    const survivalDistributionData = [
        { index: 'Survived', value: overallSurvival.survived },
        { index: 'Not Survived', value: overallSurvival.notSurvived }
    ];
    
    if (overallSurvival.survived + overallSurvival.notSurvived > 0) {
        // Try piechart with capital C (correct function name)
        if (typeof tfvis.render.piechart === 'function') {
            tfvis.render.piechart(
                { name: 'Overall Survival Distribution', tab: 'Charts' },
                survivalDistributionData,
                {
                    height: 300,
                    width: 400
                }
            );
        } else {
            // Fallback to barchart if piechart doesn't exist
            tfvis.render.barchart(
                { name: 'Overall Survival Distribution', tab: 'Charts' },
                survivalDistributionData,
                {
                    xLabel: 'Outcome',
                    yLabel: 'Count',
                    height: 300
                }
            );
        }
    }
    
    // Age distribution by survival - FIXED DATA FORMAT
    const survivedAges = trainData
        .filter(row => row.Survived === 1 && row.Age !== null && !isNaN(row.Age))
        .map(row => row.Age);
    
    const notSurvivedAges = trainData
        .filter(row => row.Survived === 0 && row.Age !== null && !isNaN(row.Age))
        .map(row => row.Age);
    
    if (survivedAges.length > 0 || notSurvivedAges.length > 0) {
        // For histogram, we need to format data differently
        const ageData = {
            values: [survivedAges, notSurvivedAges],
            series: ['Survived', 'Not Survived']
        };
        
        tfvis.render.histogram(
            { name: 'Age Distribution by Survival', tab: 'Charts' },
            ageData,
            { 
                xLabel: 'Age', 
                yLabel: 'Count',
                height: 300
            }
        );
    }
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation values from training data
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(age => age !== null));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(fare => fare !== null));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    
    // Standardize numerical features
    const standardizedAge = (age - ageMedian) / (calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null)) || 1);
    const standardizedFare = (fare - fareMedian) / (calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null)) || 1);
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass values: 1, 2, 3
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
    // Add optional family features if enabled
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    
    return features;
}

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    
    values.sort((a, b) => a - b);
    const half = Math.floor(values.length / 2);
    
    if (values.length % 2 === 0) {
        return (values[half - 1] + values[half]) / 2;
    }
    
    return values[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (values.length === 0) return null;
    
    const frequency = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create a sequential model
    model = tf.sequential();
    
    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    // Simple summary since tfjs doesn't have a built-in summary function for the browser
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'accuracy', 'val_loss', 'val_accuracy'],  // Use full metric names
                    { callbacks: ['onEpochEnd'] }
                ),
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                }
            }
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Calculate confusion matrix
    const predVals = validationPredictions.arraySync();
    const trueVals = validationLabels.arraySync();
    
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;
    
    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;
    
    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals);
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ threshold, fpr, tpr });
    });
    
    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    // Plot ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
        { 
            xLabel: 'False Positive Rate', 
            yLabel: 'True Positive Rate',
            series: ['ROC Curve'],
            width: 400,
            height: 400
        }
    );
    
    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = testPredictions.arraySync();
        
        // Create prediction results
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i]
        }));
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows - FIXED: safer type checking
    data.forEach(row => {
        const tr = document.createElement('tr');
        ['PassengerId', 'Survived', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            let value = row[key];
            
            // Handle different data types safely
            if (key === 'Probability') {
                // Ensure it's a number before using toFixed
                if (typeof value === 'number' && !isNaN(value)) {
                    td.textContent = value.toFixed(4);
                } else {
                    td.textContent = 'N/A';
                    td.style.color = '#999';
                }
            } else {
                // Convert to string for display
                td.textContent = String(value !== undefined && value !== null ? value : 'N/A');
            }
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Get predictions
        const predArray = await testPredictions.array();
        const flatPredValues = predArray.flat();
        
        // Create submission CSV with BOM (Byte Order Mark) for Excel compatibility
        let submissionCSV = '\uFEFFPassengerId,Survived\n'; // BOM for Excel
        let probabilitiesCSV = '\uFEFFPassengerId,Probability\n'; // BOM for Excel
        
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const id = preprocessedTestData.passengerIds[i];
            const prediction = flatPredValues[i];
            const survived = prediction >= 0.5 ? 1 : 0;
            
            submissionCSV += `${id},${survived}\n`;
            probabilitiesCSV += `${id},${Number(prediction).toFixed(6)}\n`;
            
            // Debug log first few rows
            if (i < 5) {
                console.log(`Row ${i}: ID=${id}, Prediction=${prediction}, Survived=${survived}`);
            }
        }
        
        console.log('Final submission CSV sample:', submissionCSV.split('\n').slice(0, 10).join('\n'));
        
        // Create and trigger downloads with explicit MIME type
        const submissionBlob = new Blob([submissionCSV], { type: 'text/csv; charset=utf-8' });
        const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv; charset=utf-8' });
        
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(submissionBlob);
        submissionLink.download = 'submission.csv';
        document.body.appendChild(submissionLink);
        submissionLink.click();
        document.body.removeChild(submissionLink);
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(probabilitiesBlob);
        probabilitiesLink.download = 'probabilities.csv';
        document.body.appendChild(probabilitiesLink);
        probabilitiesLink.click();
        document.body.removeChild(probabilitiesLink);
        
        // Save model
        await model.save('downloads://titanic-tfjs-model');
        
        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: submission.csv (Kaggle submission format)</p>
            <p>Downloaded: probabilities.csv (Prediction probabilities)</p>
            <p>Model saved to browser downloads</p>
            <p><strong>Note:</strong> If the CSV appears corrupted in Excel, try opening it in a text editor or use Google Sheets.</p>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}