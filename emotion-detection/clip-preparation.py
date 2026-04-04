import stable_whisper

# everything we need to prepare our stimulus 
# install ffmpeg (e.g., brew install ffmpeg)

def main():
    # -> mp4 to wav 
    # command: ffmpeg -i <input file> -ac 2 -f wav <output file (.wav)>

    # extracting transcript 
    # using stable-ts -> saving as json
    clip = "content/clip.mp4"
    model = stable_whisper.load_model('base')

    # downloaded the real script, made some modifications (actor improvisations)
    actual_transcript = "Hello, Rom. I’ve waited years. How many years? By now it must be... Twenty-three years, four months, eight days. Doyle? I thought I was prepared. I knew all the theory. Reality’s different. And Miller? There’s nothing here for us. Why didn’t you sleep? Oh I had a couple of stretches. But I stopped believing you were coming back, and something seems wrong about dreaming my life away. I learned what I could from studying the black hole, but I couldn’t send anything to your father. We’ve been receiving, but nothing gets out. Is he alive? Oh yea. We’ve got years of messages stored. Cooper. Messages span twenty-three years. Play it from the beginning. Hi, Dad, just checking in, saying hi. Um, finished second in school. Ms Kurling's still giving me C's though, pulled me down but second's not bad. Grandpa attended the ceremony. Uh, oh, I met another girl, Dad. I really think this is the one, her name's lois, that's her right there. Murpy stole Grandpa’s car. She crashed it, she’s okay, though. Hey Dad, look at this, you're a grandpa. His name's Jesse. I kinda wanted to call him Coop, but Lois says uh, maybe next time. Donald said he's already earned the great part so, he just leaves it at that. Oh dear, oh dear, uh oh, say bye-bye grandpa, bye-bye grandpa ok. Sorry it's been a while, just, with Jesse and all ... uh Grandpa died last week. We buried him out in the back forty, next to Mom and Jesse. Which is where we’d have buried you, if you’d ever come back. Murph was there for the funeral. We don’t see her so much but she came for that. You’re not listening to this. I know that. All these messages are just out there, drifting out there in the darkness... Lois says that uh... I have to uh... let you go and uh so, I guess, I'm letting you go. I don't know wherever you are, Dad, uh... I hope that you're at peace, and... bye. Hey, Dad. Hey Murph. You sonofabitch. I never made one of these when you were still responding cos I was so mad at you for leaving. And then when you went quiet, it seemed like I should just live with that decision. And I have... But today’s my birthday. And it’s a special one because you told me... You once told me that when you came back we might be the same age... and today I’m the age you were when you left... So it’d be a real good time for you to come back."

    # Align instead of transcribe for accuracy. The dialogue in this scene is so important, so i need enhanced accuracy.
    result = model.align(clip, actual_transcript, language='en')

    result.save_as_json("transcript.json")

    if __name__ == "__main__":
        main()