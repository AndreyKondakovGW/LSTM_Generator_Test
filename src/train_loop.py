import numpy as np

def training_loop(model,loader, num_epochs, optimizer,scheduler, criterion, device):
    loss_avg = []

    for epoch in range(num_epochs):
        model.train()
        train, target = loader.get_batch()
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        hidden = model.init_hidden(batch_size=loader.batch_size, device=device)

        output, hidden = model(train, hidden)
        loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_avg.append(loss.item())
        if len(loss_avg) >= 50:
            mean_loss = np.mean(loss_avg)
            print(f'Loss: {mean_loss}')
            scheduler.step(mean_loss)
            loss_avg = []
        """ if len(loss_avg) >= 50:
            mean_loss = np.mean(loss_avg)
            print(f'Loss: {mean_loss}')
            scheduler.step(mean_loss)
            loss_avg = []
            model.eval()
            predicted_text = evaluate(model, encoder.char_to_idx, encoder.idx_to_char)
            print(predicted_text) """


