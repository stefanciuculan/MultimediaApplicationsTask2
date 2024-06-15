1.
Folosind modelul generativ Stable Diffusion din keras-cv (puteți citi mai multe aici), generați o serie de imagini pe baza unui prompt (o propoziție). Aceste propoziții trebuie să conțină un obiect de o anumită culoare și un fundal, aceste trei caracteristici fiind atribuite fiecărui student în parte (coloanele COLOR, OBJECT și BACKGROUND din resursa ASIGNARE.pdf).
Este recomandat să experimentați cu modul de formare al promptului, astfel încât imaginile generate să reprezinte bine atât obiectul asignat cât și fundalul.
ATENȚIE: Generarea imaginilor rulează local, implicit folosindu-se procesorul, nu placa grafică. De aceea, timpul de generarea al unei imagini poate fi considerabil, în funcție de resursele disponibile.
Dacă sunteți interesați, puteți experimenta cu rularea rețelei pe GPU (dacă acest lucru este posibil pe sistemul vostru de operare).
2.
Generați o serie de imagini pe baza imaginii de la punctul anterior:
a.
Desenați un dreptunghi vizibil în jurul obiectului de interes.
b.
Aplicați o mască de culoare pentru a păstra din imaginea inițială doar ce este specificat în coloana TO_EXTRACT din ASIGNARE.pdf, și anume fie doar obiectul, fie doar fundalul.
c.
Obțineți separat masca binarizată aplicată mai sus
d.
Transformați imaginea inițială și imaginea obținută la punctul b în spațiul de culoare asignat în coloana COLOR_SPACE din ASIGNARE.pdf.
3.
Pe baza imaginii inițiale și a imaginilor obținute la punctul 2, creați un video care să folosească codecul asignat în coloana CODEC din ASIGNARE.pdf.
4.
Pentru ficare imagine din video, plasați un text vizibil care să descrie imaginea afișată (de exemplu “Original image with a red car in a forest”, “Mask with the background”, etc.).
5.
Pe baza titlurilor alese la punctul 4, folosind biblioteca de text-to-speech pyttsx3, generați un fișier audio care să conțină pronunția textului scris pe imagini. Trebuie să aveți atenție la sicronizarea timpului de pronunțare a textului cu timpul de afișare a respectivei imagini în video.
Configurări și instalări
1.
Pentru generarea de imagini, puteți folosi environment-ul virtual al laboratorului, am, la care mai trebuie să adăugați modulele tensorflow și keras_cv pip install tensorflow pip install --upgrade keras_cv
2.
Pentru salvarea sau prelucrarea imaginilor puteți folosi modulul opencv (dacă nu aveți deja creat environment-ul am, atunci vă puteți crea un environment nou care să conțină doar modulele folosite în această temă).
3.
Pentru sintetizarea fișierului audio, instalați biblioteca pyttsx3: pip install pyttsx3
