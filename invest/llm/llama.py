import torch, pdb, time
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "feature-extraction",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

batched_data = [
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
    "Big Tech Earnings Under Pressure: Major tech firms like Apple, Amazon, Meta, Microsoft, and Tesla are experiencing significant stock declines—Tesla nearly 25%—due to new tariffs and policy unpredictability under the Trump administration. These companies, heavily reliant on international supply chains, are facing intensified investor concerns. ​",
    "Apple's Manufacturing Shift: In response to tariff impacts, Apple plans to relocate all U.S.-bound iPhone manufacturing from China to India by 2026. ​",
    "IBM's Domestic Investment: IBM announced a $150 billion investment in U.S. manufacturing over the next five years, focusing on quantum computing and AI, aligning with the administration's push for domestic production.",
    "Landmark AI Legislation: The U.S. Congress passed the 'Take It Down Act,' criminalizing non-consensual deepfake pornography and mandating social media platforms to remove such content within 48 hours of notification. Tech companies like Meta and Snapchat supported the bill. ​",
    "Google's Antitrust Ruling: A significant antitrust ruling was made against Google's ad tech monopoly, marking a pivotal moment in tech regulation. ​",
]

start_time = time.perf_counter()
data = pipe(batched_data[:20])
end_time = time.perf_counter()
print(f"processed {len(batched_data)} sentences")
print(f"Time taken: {end_time - start_time:.4f} seconds")
pdb.set_trace()
print(data)

"""

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a funny friend and always makes jokes with your baby cousin who is a university undergrad!"},
    {"role": "user", "content": "try to say this in a funny way to you baby cousin: How is life and how is research? Are you working hard? Don't be cause your big bro wants you to be happy and spend more fun time with us in California."},
]
outputs = pipe(
    messages,
    max_new_tokens=2560,
)
print(outputs[0]["generated_text"][-1])
"""

