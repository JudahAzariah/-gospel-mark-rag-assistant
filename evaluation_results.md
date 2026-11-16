# Lab 2 Retrieval Evaluation

Top-k: 3

## Test 1: Who baptized Jesus?
- Expected evidence: Mark 1:9 (John the Baptist)
- Result: **HIT**
- Retrieved chunks:
  - Top 1: `corpus/mark.txt` / `MARK-200-001`, similarity `0.5528`, refs: Mark 1:1, Mark 1:2, Mark 1:3, Mark 1:4, Mark 1:5, Mark 1:6, Mark 1:7, Mark 1:8, Mark 1:9, Mark 1:10
    - Snippet: Mark 1:1 | The beginning of the gospel of Jesus Christ, the Son of God; Mark 1:2 | As it is written in the prophets, Behold, I send my messenger before thy face, which shall prepare thy way before thee. Mark 1:3 | The voice of one crying in the wilderness, Pre
  - Top 2: `corpus/mark.txt` / `MARK-200-002`, similarity `0.5013`, refs: Mark 1:9, Mark 1:10, Mark 1:11, Mark 1:12, Mark 1:13, Mark 1:14, Mark 1:15, Mark 1:16, Mark 1:17, Mark 1:18, Mark 1:19
    - Snippet: Mark 1:9 | And it came to pass in those days, that Jesus came from Nazareth of Galilee, and was baptized of John in Jordan. Mark 1:10 | And straightway coming up out of the water, he saw the heavens opened, and the Spirit like a dove descending upon him: Mark 
  - Top 3: `corpus/mark.txt` / `MARK-200-058`, similarity `0.4907`, refs: Mark 10:39, Mark 10:40, Mark 10:41, Mark 10:42, Mark 10:43, Mark 10:44, Mark 10:45, Mark 10:46
    - Snippet: Mark 10:39 | And they said unto him, We can. And Jesus said unto them, Ye shall indeed drink of the cup that I drink of; and with the baptism that I am baptized withal shall ye be baptized: Mark 10:40 | But to sit on my right hand and on my left hand is not mi

## Test 2: What did Jesus say to the storm?
- Expected evidence: Mark 4:39 (Peace, be still)
- Result: **HIT**
- Retrieved chunks:
  - Top 1: `corpus/mark.txt` / `MARK-200-020`, similarity `0.5193`, refs: Mark 4:35, Mark 4:36, Mark 4:37, Mark 4:38, Mark 4:39, Mark 4:40, Mark 4:41, Mark 5:1, Mark 5:2
    - Snippet: Mark 4:35 | And the same day, when the even was come, he saith unto them, Let us pass over unto the other side. Mark 4:36 | And when they had sent away the multitude, they took him even as he was in the ship. And there were also with him other little ships. Ma
  - Top 2: `corpus/mark.txt` / `MARK-200-059`, similarity `0.5066`, refs: Mark 10:45, Mark 10:46, Mark 10:47, Mark 10:48, Mark 10:49, Mark 10:50, Mark 10:51, Mark 10:52
    - Snippet: Mark 10:45 | For even the Son of man came not to be ministered unto, but to minister, and to give his life a ransom for many. Mark 10:46 | And they came to Jericho: and as he went out of Jericho with his disciples and a great number of people, blind Bartimaeus
  - Top 3: `corpus/mark.txt` / `MARK-200-023`, similarity `0.5061`, refs: Mark 5:15, Mark 5:16, Mark 5:17, Mark 5:18, Mark 5:19, Mark 5:20, Mark 5:21, Mark 5:22, Mark 5:23
    - Snippet: Mark 5:15 | And they come to Jesus, and see him that was possessed with the devil, and had the legion, sitting, and clothed, and in his right mind: and they were afraid. Mark 5:16 | And they that saw it told them how it befell to him that was possessed with th

## Test 3: How many loaves were available when Jesus fed the five thousand?
- Expected evidence: Mark 6:38 (Five loaves)
- Result: **HIT**
- Retrieved chunks:
  - Top 1: `corpus/mark.txt` / `MARK-200-032`, similarity `0.6190`, refs: Mark 6:34, Mark 6:35, Mark 6:36, Mark 6:37, Mark 6:38, Mark 6:39, Mark 6:40, Mark 6:41
    - Snippet: Mark 6:34 | And Jesus, when he came out, saw much people, and was moved with compassion toward them, because they were as sheep not having a shepherd: and he began to teach them many things. Mark 6:35 | And when the day was now far spent, his disciples came un
  - Top 2: `corpus/mark.txt` / `MARK-200-040`, similarity `0.5587`, refs: Mark 7:37, Mark 8:1, Mark 8:2, Mark 8:3, Mark 8:4, Mark 8:5, Mark 8:6, Mark 8:7, Mark 8:8
    - Snippet: Mark 7:37 | And were beyond measure astonished, saying, He hath done all things well: he maketh both the deaf to hear, and the dumb to speak. Mark 8:1 | In those days the multitude being very great, and having nothing to eat, Jesus called his disciples unto hi
  - Top 3: `corpus/mark.txt` / `MARK-200-033`, similarity `0.5505`, refs: Mark 6:41, Mark 6:42, Mark 6:43, Mark 6:44, Mark 6:45, Mark 6:46, Mark 6:47, Mark 6:48, Mark 6:49, Mark 6:50
    - Snippet: Mark 6:41 | And when he had taken the five loaves and the two fishes, he looked up to heaven, and blessed, and brake the loaves, and gave them to his disciples to set before them; and the two fishes divided he among them all. Mark 6:42 | And they did all eat, 

## Test 4: Who betrayed Jesus?
- Expected evidence: Mark 14:10 (Judas Iscariot)
- Result: **MISS**
- Retrieved chunks:
  - Top 1: `corpus/mark.txt` / `MARK-200-083`, similarity `0.5632`, refs: Mark 14:40, Mark 14:41, Mark 14:42, Mark 14:43, Mark 14:44, Mark 14:45, Mark 14:46, Mark 14:47, Mark 14:48
    - Snippet: Mark 14:40 | And when he returned, he found them asleep again, (for their eyes were heavy,) neither wist they what to answer him. Mark 14:41 | And he cometh the third time, and saith unto them, Sleep on now, and take your rest: it is enough, the hour is come; 
  - Top 2: `corpus/mark.txt` / `MARK-200-091`, similarity `0.5257`, refs: Mark 15:35, Mark 15:36, Mark 15:37, Mark 15:38, Mark 15:39, Mark 15:40, Mark 15:41, Mark 15:42, Mark 15:43
    - Snippet: Mark 15:35 | And some of them that stood by, when they heard it, said, Behold, he calleth Elias. Mark 15:36 | And one ran and filled a spunge full of vinegar, and put it on a reed, and gave him to drink, saying, Let alone; let us see whether Elias will come to
  - Top 3: `corpus/mark.txt` / `MARK-200-084`, similarity `0.5122`, refs: Mark 14:47, Mark 14:48, Mark 14:49, Mark 14:50, Mark 14:51, Mark 14:52, Mark 14:53, Mark 14:54, Mark 14:55, Mark 14:56, Mark 14:57
    - Snippet: Mark 14:47 | And one of them that stood by drew a sword, and smote a servant of the high priest, and cut off his ear. Mark 14:48 | And Jesus answered and said unto them, Are ye come out, as against a thief, with swords and with staves to take me? Mark 14:49 | 

## Test 5: What was the name of the blind man Jesus healed near Jericho?
- Expected evidence: Mark 10:46 (Bartimaeus)
- Result: **HIT**
- Retrieved chunks:
  - Top 1: `corpus/mark.txt` / `MARK-200-059`, similarity `0.5930`, refs: Mark 10:45, Mark 10:46, Mark 10:47, Mark 10:48, Mark 10:49, Mark 10:50, Mark 10:51, Mark 10:52
    - Snippet: Mark 10:45 | For even the Son of man came not to be ministered unto, but to minister, and to give his life a ransom for many. Mark 10:46 | And they came to Jericho: and as he went out of Jericho with his disciples and a great number of people, blind Bartimaeus
  - Top 2: `corpus/mark.txt` / `MARK-200-043`, similarity `0.5189`, refs: Mark 8:22, Mark 8:23, Mark 8:24, Mark 8:25, Mark 8:26, Mark 8:27, Mark 8:28, Mark 8:29, Mark 8:30, Mark 8:31
    - Snippet: Mark 8:22 | And he cometh to Bethsaida; and they bring a blind man unto him, and besought him to touch him. Mark 8:23 | And he took the blind man by the hand, and led him out of the town; and when he had spit on his eyes, and put his hands upon him, he asked h
  - Top 3: `corpus/mark.txt` / `MARK-200-091`, similarity `0.4820`, refs: Mark 15:35, Mark 15:36, Mark 15:37, Mark 15:38, Mark 15:39, Mark 15:40, Mark 15:41, Mark 15:42, Mark 15:43
    - Snippet: Mark 15:35 | And some of them that stood by, when they heard it, said, Behold, he calleth Elias. Mark 15:36 | And one ran and filled a spunge full of vinegar, and put it on a reed, and gave him to drink, saying, Let alone; let us see whether Elias will come to

**Hit rate: 4/5 = 80%**
