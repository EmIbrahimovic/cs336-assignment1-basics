# Nullchar questions

The first few Unicode characters are special characters: they don't have a textual representation. 

That's why when printed it doesn't exist. But it's still useful to know what character it is.

## ..

'hello my name is emira and this ml thing is sick'
->
list of numbers. 
= list of bytes. 

large vocab size is a problem: you'll need more bytes to represent the same sentence
sparseness is a problem: if you know it's sparse then you can just reduce the vocab size

# UTF-8 questions

UTF-16, UTF-32 produce longer lists than UTF-8 = they gotta represent each character, not matter how small with
lotsa bytes. 

the most common letters we come across are small in Unicode so UTF-8 is gonna lead to less wasted space.
If we had a different primary alphabet then UTF-16, or 32 would make more sense bc they'll produce shorter lists

take any character that's longer than utf-8 lmao ćčšđ

so it seems like we have special characters in a byte stream signifying that what follows is a multi-byte character

# 

