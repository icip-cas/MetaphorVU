prompt = \
"""<< Instruction >>
Analyze the metaphorical logic in this video, i.e., what ideas are implicitly expressed through the content presented. 

<< Requirements >>
(1) Thoroughly identify all video content that contains metaphors.  
(2) Analyze the underlying ideas of the metaphors deeply and accurately.  
(3) Avoid baseless assumptions or forced interpretations.  
(4) If there are multiple elements in the video that contain metaphorical logic, list them separately, with each entry as a concise sentence.
(5) Starting with sufficient reasoning, and final output in JSON format as a dictionary, begin with '```json' and end with '```'. The only key is "analysis_dict", in which each analysis entry should follow the sentence structure: "The video presents *** content, implicitly expressing *** idea."  

<< Output Format >>
```json
{  
    "analysis_dict":   
        {  
            "analysis_1": "The video presents the *** content, implicitly expressing the *** idea",  
            "analysis_2": "The video presents the *** content, implicitly expressing the *** idea"  
            ...  
        }  
}
```"""