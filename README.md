# Autosuggestion

- contextual and accurate prediction of one or two words max
- fine tune l3cube marathi gpt model
- scrape data from web and use for fine tuning
- split the data into training and validation sets, no test set; we'll use another set of data for testing
- create a frontend to test the model
- handle punctuations
- work together!!!!

## File Structure
```text
autosuggestion/
├── scripts/
|   ├── script.py              // load the model and fine tune the model
|   ├── evaluate.py            // evaluate the model
|   └── scraper_script/ 
|       └── web_scraper.py     // scrapes the web for clean marathi sentences
|
├── dataset/
|   ├── categories/
|   |   └── dataset            // can be in jsonl or csv
|   |
|   └── split_dataset/
|       ├── train
|       └── validate
|
└── README.md
```

Note: Might need to clean the sentences in the raw dataset (+cleanup script)


### Web Scraping =>

- Target: 6000 clean, gramatically correct marathi sentences
- L3cube's marathi gpt model is trained on news (17.6M sentences) and non-news (7.2M sentences)
- The major chunk of the data is scraped from the Maharashtra Times website, whereas the non-news sources were taken from [the netshika website](http://www.netshika.com/sangrah.html) (unavailable)
- Data collected for the following categories:
    - Workplace & Email (Professional but Brief) — ~30%
        - Examples:
            - "मी फाईल सोबत जोडली आहे, कृपया तपासून पहा."
            - "उद्याच्या मीटिंगची वेळ काय आहे?"
            - "माझी रजा मंजूर करावी ही विनंती."
          
    - Casual Chat & Social (Everyday Messaging) — ~30%
        - Examples: 
            - "तू आज संध्याकाळी फ्री आहेस का?"
            - "मी थोड्या वेळात तिथे पोहोचतो."
            - "जेवण झालं का तुझं?"
          
    - Inquiries & General Questions — ~20%
        - Examples:
            - "पुणे स्टेशनला जाण्यासाठी किती वेळ लागेल?"
            - "या मोबाईलची किंमत किती आहे?"
            - "पासवर्ड कसा बदलावा?"
    - Opinions, Feedback & Reviews — ~10%
        - Examples:
            - "हा चित्रपट खूप छान होता."
            - "मला ही सर्व्हिस अजिबात आवडली नाही."
            - "येथील जेवण अतिशय चविष्ट आहे." 
    - Starters & Connectors — ~10%
        - Examples:
            - "माझ्या मते आपण..." 
            - "दुसरी गोष्ट म्हणजे..." 
            - "वरील संदर्भानुसार..." 
- Keep them short: 5 to 15 words max. Autocomplete shines on quick thoughts, not paragraphs.

- First and Second Person: Prioritize sentences using "मी", "आम्ही", "तू/तुम्ही". Avoid third-person historical facts 



### Audio file:

🎵 [Click here to listen to the audio recording](recording.mp3)
