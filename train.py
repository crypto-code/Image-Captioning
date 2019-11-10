import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import loader
from build_vocab import Vocabulary
from model import EncoderCNN , DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
	#creating model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	transform = transforms.Compose([
		transforms.RandomCrop(args.crop_size),
		transforms.RandomHorizontalFlip(), 
		transforms.ToTensor(), 
		transforms.Normalize((0.485, 0.456, 0.406), 
							 (0.229, 0.224, 0.225))])
	
	#load the vocabulary wrapper
	with open(args.vocab_path, 'rb') as f:
		vocab = pickle.load(f)

	#build data loader
	data_loader = loader(args.image_dir, args.caption_path, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

	#build the models
	encoder = EncoderCNN(args.embed_size).to(device)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

	#loss and optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)

	#training the model	
	total_step = len(data_loader)
	for epoch in range(args.num_epochs):
		for i, (images, captions, lengths) in enumerate(data_loader):
			images = images.to(device)
			captions = captions.to(device)
			targets = pack_padded_sequence(captions, lengths, batch_first = True)[0]

			#forward, backward and optimize
			features = encoder(images)
			outputs = decoder(features, captions, lengths)
			loss = criterion(outputs, targets)
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()

			# Print log info
			if i % args.log_step == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
					  .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

			if i%args.save_step ==0:
                                torch.save(decoder.state_dict(), os.path.join(args.model_path, 'res_decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                                torch.save(encoder.state_dict(), os.path.join(args.model_path, 'res_encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
				
	
					


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='data/build/models/' , help='path for saving trained models')
	parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
	parser.add_argument('--vocab_path', type=str, default='data/build/vocab.pkl', help='path for vocabulary wrapper')
	parser.add_argument('--image_dir', type=str, default='data/resized2014/', help='directory for resized images')
	parser.add_argument('--caption_path', type=str, default='data/train2014/captions_train2014.json', help='path for train annotation json file')
	parser.add_argument('--log_step', type=int , default=1, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=40, help='step size for saving trained models')
	
	# Model parameters
	parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
	
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	args = parser.parse_args()
	print(args)
	main(args)