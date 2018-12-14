# Optical Music Recognition
This algorithm reads written music notes from note sheet images and converts them into musical outputs.
It is also able to able to read notes from non-uniformly illuminated, crumpled, old sheets and also from skewed sheet images with an accuracy around 90%.

*findNotes* applies homomorphic filtering, binarization and skew correction to the image. After these initial operations it is easier to locate staff lines on the image. Once staff lines are located, *findNotes* calculates the position of each note in terms of previosuly located staff lines and returns these positions as output. *findNotes* also locates some of the musical signs like clef using correlation matrices.

*playNotes* simply constitutes sinusoidals matching found notes in previous stage and plays the musical output.

The algorithm is successfull in simple music sheets like Jingle Bells image but for complex music sheets, it is not possible to locate all note heads and classify each of them correctly. At this point, algorithm needs to be improved. 
