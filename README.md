# 🚗 Automotive Appointment Booking AI

An intelligent AI system that learns to book appointments with potential car buyers using reinforcement learning.

## 🎯 What This Does

This AI learns to have conversations with potential car buyers and convince them to book appointments at your dealership. It learns through trial and error, getting better over time.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the AI
```bash
cd salesgym/src
python train.py
```

### 3. Watch the AI in Action! 🎭
```bash
cd salesgym/src
python conversation_simulator.py
```

### 4. Easy Business Interface
```bash
cd salesgym/src
python business_interface.py
```

## 🎭 What You'll See

The AI will have **real conversations** like this:

```
👤 Customer: Cautious Buyer
📊 Initial State: Interest=0.35, Trust=0.25, Commitment=0.20

🤖 AI: Hi! I'm calling about the car you're interested in. How are you doing today?
📊 Customer State: Interest=0.45, Trust=0.35, Commitment=0.24

🤖 AI: What kind of vehicle are you looking for?
📊 Customer State: Interest=0.52, Trust=0.40, Commitment=0.26

🤖 AI: I can get you a great deal on this vehicle. Let me see what discounts I can apply.
📊 Customer State: Interest=0.58, Trust=0.44, Commitment=0.34

🤖 AI: Would you like to schedule a test drive? I have some time slots available.
✅ SUCCESS: Customer booked an appointment!
```

## ⚙️ Easy Configuration

You can customize the AI's behavior by editing `config.json`. No coding required!

### Customer Psychology Settings

Adjust how customers behave:

```json
{
  "customer_psychology": {
    "interest": {
      "description": "How interested the customer is in buying a car",
      "ready_threshold": 0.5
    },
    "trust": {
      "description": "How much the customer trusts the dealership", 
      "ready_threshold": 0.5
    }
  }
}
```

### Sales Actions

Customize how each sales tactic affects customers:

```json
{
  "sales_actions": {
    "rapport": {
      "description": "Build relationship and trust with customer",
      "effects": {
        "interest": 0.10,
        "trust": 0.06,
        "commitment": 0.03
      }
    }
  }
}
```

### Customer Types

Define different types of customers:

```json
{
  "customer_types": {
    "type_0": {
      "name": "Cautious Buyer",
      "description": "Takes time to decide, needs lots of information",
      "base_psychology": {
        "interest": 0.30,
        "trust": 0.25,
        "commitment": 0.20
      }
    }
  }
}
```

## 🎭 Sales Actions Explained

| Action | What It Does | Best For |
|--------|-------------|----------|
| **Rapport** | Builds relationship and trust | Starting conversations |
| **Qualify** | Learns about customer needs | Understanding what they want |
| **Show Inventory** | Shows available cars | When they're interested |
| **Handle Concerns** | Addresses price/reliability worries | When they're hesitant |
| **Create Urgency** | Creates time pressure | When they're ready but slow |
| **Social Proof** | Shows others bought and are happy | Building confidence |
| **Offer Incentives** | Offers special deals/financing | Closing the deal |
| **Book Appointment** | Tries to book the appointment | When they're ready |

## 📊 Understanding Results

### Training Results
- **Win Rate**: Percentage of successful appointments booked
- **Action Distribution**: Which tactics the AI uses most often

### Example Output
```
Win rate over 500 episodes: 0.982

Action distribution:
  Action 0:  366 (  8.2%)  # Rapport
  Action 1:  298 (  6.7%)  # Qualify  
  Action 2:  466 ( 10.4%)  # Show Inventory
  Action 3:  365 (  8.2%)  # Handle Concerns
  Action 5:  798 ( 17.9%)  # Social Proof
  Action 6: 1833 ( 41.1%)  # Offer Incentives
  Action 7:  334 (  7.5%)  # Book Appointment
```

## 🔧 Advanced Configuration

### Environment Settings
```json
{
  "environment": {
    "max_conversation_turns": 12,
    "num_customer_types": 6,
    "success_threshold": 0.6
  }
}
```

### Customizing Action Effects

To make an action more effective, increase its values:

```json
{
  "sales_actions": {
    "offer_incentives": {
      "effects": {
        "commitment": 0.12  # Increased from 0.08
      }
    }
  }
}
```

### Adding New Customer Types

```json
{
  "customer_types": {
    "type_6": {
      "name": "Luxury Buyer",
      "description": "High-end customer, price not a concern",
      "base_psychology": {
        "interest": 0.70,
        "urgency": 0.30,
        "availability": 0.90,
        "trust": 0.60,
        "commitment": 0.50
      }
    }
  }
}
```

## 🎯 Business Use Cases

### 1. **Training Sales Staff**
- Show them which tactics work best
- Practice with different customer types
- Learn optimal conversation flow

### 2. **A/B Testing Sales Strategies**
- Test different action effects
- Compare success rates
- Optimize for your specific market

### 3. **Customer Segmentation**
- Understand different customer types
- Tailor approaches for each segment
- Improve conversion rates

## 📈 Performance Tips

### High Success Rate (>90%)
- Your AI is well-trained
- Consider making customers more challenging
- Try different customer type distributions

### Low Success Rate (<70%)
- Increase training time
- Check action effects are realistic
- Verify customer thresholds make sense

### Unbalanced Action Usage
- One action used too much: Reduce its effectiveness
- One action never used: Increase its effectiveness
- All actions used equally: Add more variety to customer types

## 🛠️ Troubleshooting

### Common Issues

**AI always uses the same action:**
- Check if one action is too effective
- Reduce its values in config.json

**AI never books appointments:**
- Lower the success thresholds
- Increase commitment values for actions

**Training is too slow:**
- Reduce max_conversation_turns
- Use fewer customer types initially

### Getting Help

1. Check the configuration file is valid JSON
2. Verify all required fields are present
3. Test with a simple configuration first
4. Check the logs for error messages

## 📁 File Structure

```
salesgym/
├── src/
│   ├── purchase_env.py      # Main environment
│   ├── config.py            # Configuration system
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   ├── smoke_test.py        # Test script
│   └── config.json          # Configuration file
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Technical Notes

### Environment Details
- **Environment**: `AutomotiveAppointmentEnv` with discrete actions (sales tactics) and customer psychology features
- **Reward**: Sparse reward - only at episode end (1 if appointment booked, 0 otherwise)
- **Actions**: 8 different sales tactics (rapport, qualify, show_inventory, etc.)
- **Customer Features**: 5 psychological traits (interest, urgency, availability, trust, commitment)

### Customization
- Tune `action_effects` in `config.json` to shape task difficulty
- Adjust customer psychology thresholds for different markets
- Modify persona priors for different customer segments

## 🎉 Success!

Your AI is now ready to learn how to book automotive appointments! The system will automatically adapt to your configuration and get better over time.

Remember: The AI learns through trial and error, so the more you train it, the better it gets!