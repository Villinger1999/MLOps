import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(10):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        
    print("Training complete")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = corrupt_mnist()
    
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).sum().item()
            total += target.size(0)
    print(f"Test accuracy: {correct / total}")
    

if __name__ == "__main__":
    app()
