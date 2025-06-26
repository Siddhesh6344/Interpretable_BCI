
from tqdm import tqdm

# ---------------------------
# Training Loop
# ---------------------------

def train(model, dataloader, optimizer, criterion, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)
        for data, _ in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, pre_codes, codes, dictionary = model(data)
            dictionary = model.encoder.weight.view(1000, -1).detach()  # [1000, 784]
            loss = criterion(data, x_hat, pre_codes, codes, dictionary)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {epoch_loss / len(dataloader):.6f}")
