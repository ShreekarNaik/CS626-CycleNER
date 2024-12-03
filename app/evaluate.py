import torch

def prepare_data(sentences, entity_sequences, tokenizer, max_length=512):
    inputs = tokenizer(sentences, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    outputs = tokenizer(entity_sequences, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return inputs.input_ids, inputs.attention_mask, outputs.input_ids

def train_model(model, tokenizer, train_sentences, train_entities, val_sentences, val_entities, 
                epochs=3, batch_size=8, learning_rate=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_inputs, train_masks, train_labels = prepare_data(train_sentences, train_entities, tokenizer)
    val_inputs, val_masks, val_labels = prepare_data(val_sentences, val_entities, tokenizer)

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_masks, val_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = (b.to(device) for b in batch)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}")

    return model
