import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils import Preprocess

# ----------- Custom LSTM from scratch (PyTorch) -----------
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combine gates for efficiency
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            CustomLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])

    def forward(self, x, states=None):
        # x shape: [batch, seq_len, input_size]
        batch, seq_len, _ = x.size()
        if states is None:
            h = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
            c = [x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = states

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(x_t, (h[layer], c[layer]))
                x_t = h[layer]
            outputs.append(h[-1].unsqueeze(1))

        out_seq = torch.cat(outputs, dim=1)
        return out_seq, (h, c)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = CustomLSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out_seq, _ = self.lstm(x)
        last = out_seq[:, -1, :]
        logits = self.fc(last)
        probs = torch.softmax(logits, dim=1)
        return probs

# ----------- Market Predictor App -----------
class MarketPredictor:
    def main(full_data, selected_df=None):
        st.header("Predicting price movement")
        st.subheader("Nepse Data")
        st.dataframe(full_data.tail(5))

        # 1. Feature extraction
        with st.spinner('Extracting features...'):
            preprocess, label, test_df = Preprocess.preprocessData(full_data)
            preprocess['Date'] = full_data['Date']
        st.success("Features extracted successfully!")

        # 2. Plot last 60
        test_df['Predicted'] = test_df['label']
        df60 = test_df[-60:]
        # (plot code same as before)
        # ...

        # 3. Feature selection
        with st.spinner('Selecting Important Features...'):
            selected_features, feature_score = Preprocess.get_imp_features_signal(preprocess, label)
        st.dataframe(selected_features.tail(5))
        fig = px.bar(x=feature_score['score'], y=feature_score['feature'], orientation='h', title='Feature Importance')
        st.plotly_chart(fig)

        # 4. Prepare data
        feat = selected_features.drop(['Date','label'], axis=1).values
        target = selected_features['label'].values.astype(int)
        # Train-test split
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            feat, target, test_size=0.1, shuffle=False
        )
        # reshape for LSTM: seq_len=1
        X_train_np = X_train_np.reshape(-1,1,X_train_np.shape[1])
        X_test_np = X_test_np.reshape(-1,1,X_test_np.shape[1])

        # Torch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.long)

        batch_size = 64
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

        # 5. Model init
        input_size = X_train.shape[2]
        hidden_size = 200
        num_layers = 1
        output_size = 3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = Generator(input_size, hidden_size, num_layers, output_size).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 6. Train
        epochs = st.slider("Select epochs", 1, 200, 50)
        with st.spinner('Training model...'):
            generator.train()
            for epoch in range(epochs):
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    preds = generator(Xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
        st.success("Model trained successfully!")

        # 7. Predict & visualize
        generator.eval()
        with torch.no_grad():
            preds = generator(X_test.to(device)).cpu().numpy()
        pred_idx = preds.argmax(axis=1)

        # Build DataFrame for plotting
        pred_df = pd.DataFrame({
            'Date': full_data['Date'].iloc[-len(pred_idx):].values,
            'Close': full_data['Close'].iloc[-len(pred_idx):].values,
            'Predicted': pred_idx
        })
        # (plot code same as before)
        # ...

        # 8. Metrics
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        cm = confusion_matrix(y_test_np, pred_idx)
        acc = accuracy_score(y_test_np, pred_idx)
        prec = precision_score(y_test_np, pred_idx, average='weighted')
        rec = recall_score(y_test_np, pred_idx, average='weighted')
        f1 = f1_score(y_test_np, pred_idx, average='weighted')

        st.subheader("Confusion Matrix & Metrics")
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['buy','sell','hold'], y=['buy','sell','hold'], colorscale='Blues'))
        st.plotly_chart(fig_cm)
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1-score: {f1:.4f}")

# To run in Streamlit:
# if __name__ == '__main__':
#     df = pd.read_csv('your_nepse_data.csv')
#     MarketPredictor.main(df)
