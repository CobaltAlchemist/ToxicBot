import discord
import discord.utils as dutil
import os
import numpy as np
from random import choice, random
from classifypt import classify
from tokenizept import tokenize
from database import ToxicDatabase

client = discord.Client(activity=discord.Game(name="?stats or ?why"))
labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult','identity_hate']

responses = [
	"Bro chill",
	"Hey that's uncalled for!",
	"Lol good one",
	"Be kind to others",
	"Why are you so mean?",
	"Try to be better",
]

apology_responses = [
	"It's ok I forgive you",
	"You went a little too far this time... but I forgive you",
	"Ok.",
	"It's alright buddy, we all have bad days",
	"I don't forgive you, but I'm proud of you for accepting responsibility",
	"Sorries are cheap, show me with your actions",
	"I know you're better than that, so I'll overlook this for now",
]

bot_responses = {
	'Dad Bot#2189': [
		"Literally do anything else other than select a random response when seeing the word 'play'",
		"Nobody asked you, dad",
	],
	'Mom Bot#8900' : [
		"AHAHAHA GET IT BECAUSE SHE'S A BOOMER THAT'S FUNNY",
		"Literally no one:\nMom bot: ^",
		"Nobody is talking to you, mom bot",
		"You're a waste of electricity and a blight on this planet, mombot",
	],
	'Dom Toretto#5330' : [
		"xd get it because family",
		"shut up you useless advertisement",
		"Hey dom bot, Family",
	]
}

user_responses = {
	'Ew0k#8394': [
		"Whatever, manlet",
		"lol short people",
		"imagine having to look up at people to talk to them",
		"keep your vacuum cleaner away from this guy",
	]
}

db = ToxicDatabase('toxicbot.csv')

memory = {}

def get_member_stats(guild, member):
	tag = f"{str(member.name)}#{str(member.discriminator)}"
	print(f"Stats asked with guild {guild} and tag {tag}")
	stats = db.user_stats(guild, tag)
	if stats['count'] == 0:
		return None
	categorical_incidents = '\n\t\t'.join([str(k) + ': ' + str(v) for k, v in zip(labels, stats['sum'])])
	return f"For user {tag} I found the following:\n"\
		   f"\tToxic incidents: {stats['count']}\n" \
		   f"\tFavorite form of toxicity (weighted): {stats['top_cat_weighted']}\n" \
		   f"\tIncidents by category:\n\t\t{categorical_incidents}"
		   
def get_server_stats(guild):
	stats = db.server_stats(guild)
	server_stats = stats[guild]
	if server_stats['count'] == 0:
		return None
	categorical_incidents = '\n\t\t'.join([str(k) + ': ' + str(v) for k, v in zip(labels, server_stats['sum'])])
	most_toxic = server_stats['most_toxic'][0]
	return f"For this server, I found the following:\n"\
		   f"\tToxic incidents: {server_stats['count']}\n" \
		   f"\tFavorite form of toxicity (weighted): {server_stats['top_cat_weighted']}\n" \
		   f"\tMost toxic member: {most_toxic[0]} with {most_toxic[1]} incidents\n" \
		   f"\tIncidents by category:\n\t\t{categorical_incidents}"
		   
def get_key(guild, author, channel):
	return guild + author + channel
	
def add_memory(guild, author, channel, message, rating):
	key = get_key(guild, author, channel)
	memory[key] = (message, rating)
	
def get_memory(guild, author, channel):
	key = get_key(guild, author, channel)
	if key not in memory:
		return None
		
	message, rating = memory[key]
	if len(message) > 100:
		message = message[:100] + '...'
		
	ratings = ',\n'.join([
		'\t' + label.replace('_', ' ').title() + ": " + str(rate)
		for label, rate in zip(labels, rating)
	])
		
	return f"Your last message, {author}: '{message}'\n" \
		   f"was rated:\n{ratings}"

@client.event
async def on_ready():
	print('We have logged in as {0.user}'.format(client))
	
@client.event
async def on_message(message):
	try:
		if message.author == client.user:
			if str(message.author) in user_responses and random() > 0.02:
				print(f"User {message.author} seen")
				print(f"Bot response sent")
				await message.channel.send(choice(user_responses[str(message.author)]))
			return
		if message.author.bot:
			print(f"Bot {message.author} seen")
			if str(message.author) in bot_responses and random() > 0.2:
				print(f"Bot response sent")
				await message.channel.send(choice(bot_responses[str(message.author)]))
			return
		if len(message.content) == 0:
			return
		
		s_guild = str(message.guild)
		s_author = str(message.author)
		
		s = message.content
		if s.lower().startswith("?stats"):
			if len(message.mentions) > 0:
				member = message.mentions[0]
				response = get_member_stats(s_guild, member)
				if response is None:
					await message.channel.send("Sorry, I didn't find any information on them")
				else:
					await message.channel.send(response)
			elif s == "?stats":
				print(f"Stats asked for guild {s_guild}")
				response = get_server_stats(s_guild)
				if response is None:
					await message.channel.send("Sorry, I didn't find any information on this server")
				else:
					await message.channel.send(response)
		elif s.lower().startswith("?why"):
			response = get_memory(s_guild, s_author, str(message.channel))
			if response is None:
				await message.channel.send("Sorry, I didn't find your last message")
			else:
				await message.channel.send(response)
		elif s.lower().startswith("?tokens "):
			if s == "?tokens ":
				await message.channel.send("Huh? Try ?tokens <type> <message>")
			type = s.split()[1]
			s_toks = s[len("?tokens ") + len(type) + 1:]
			print(f"Tokenizing '{s_toks}' with '{type}'")
			response = tokenize(s_toks, type)
			if response is None:
				await message.channel.send("Sorry, the given type was not found")
			else:
				await message.channel.send(','.join(response))
		elif s.lower().startswith("?") and len(s) > 1 and not s[1] in " ?":
			await message.channel.send("Unknown command, try '?stats' for server stats or mention a user for their stats")
			
		preds = classify(s) | classify(s.lower()) | classify(s.upper())
		print(s, '->', preds)
		if np.sum(preds) > 0:
			db.add(s_guild, s_author, preds)
		if "sorry" in s.lower() and "toxicbot" in s.lower():
			await message.channel.send(choice(apology_responses))
		if np.sum(preds) >= 4:
			await message.channel.send(choice(responses))
		elif np.sum(preds) >= 2 and preds[5] > 0:
			await message.add_reaction('<pepocringe:746992858535297044>')
		elif np.sum(preds) >= 2:
			await message.add_reaction('<cathands:755262972204679279>')
		add_memory(s_guild, s_author, str(message.channel), s, preds)
	except Exception as e:
		await message.channel.send(str(e))
		
@client.event
async def on_reaction_add(reaction, user):
	message = reaction.message
	if message.author != client.user:
		return
		
	s_guild = str(message.guild)
	s_author = str(message.author)
	
	print(f"Got reaction {reaction.emoji} to my message")
	
	if reaction.emoji in ['🎖', '🏅', '🥇'] and random() < 0.05:
		await message.channel.send('Thanks for the gold, kind stranger!')

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Toxic bot runner')
	parser.add_argument('-k', '--key', help='API Key', default=None)
	args = parser.parse_args()
	client.run(args.key)