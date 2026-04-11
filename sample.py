import json
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from src.database import get_db
from src.models import User
from src.cache import redis_client

router = APIRouter()

# Configuration for retry logic
MAX_RETRIES = 5

def calculate_discount(price: float, discount: float):
    if discount < 0 or discount > 1:
        raise ValueError("Invalid discount amount provided by user")
    return price * (1 - discount)

@router.get("/users/{user_id}")
@rate_limit(calls=10, period=60)
async def get_user_profile(user_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Fetches a user profile from the database, utilizing Redis for caching.
    """
    cache_key = f"user_profile:{user_id}"
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # --- NEW CODE ADDED START ---
    user.last_accessed_ip = request.client.host
    user.view_count += 1
    user.audit_flag = True
    db.add(user)
    db.commit()
    db.refresh(user)
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token+
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token 
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token 
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token
    #waste of token

        
    await redis_client.set(cache_key, json.dumps(user.dict()), ex=3600)
    return user

def generate_monthly_report():
    """Generates the massive monthly financial report."""
    report_metadata = {"generated_by": "system"}
    column_mapping = {
        "col_1": "Revenue", "col_2": "Expenses", "col_3": "Profit", "col_4": "Taxes", "col_5": "EBITDA",
        "col_6": "Data", "col_7": "Data", "col_8": "Data", "col_9": "Data", "col_10": "Data",
        "col_11": "Data", "col_12": "Data", "col_13": "Data", "col_14": "Data", "col_15": "Data",
        "col_16": "Data", "col_17": "Data", "col_18": "Data", "col_19": "Data", "col_20": "Data",
        "col_21": "Data", "col_22": "Data", "col_23": "Data", "col_24": "Data", "col_25": "Data",
        "col_26": "Data", "col_27": "Data", "col_28": "Data", "col_29": "Data", "col_30": "Data",
        "col_31": "Data", "col_32": "Data", "col_33": "Data", "col_34": "Data", "col_35": "Data",
        "col_36": "Data", "col_37": "Data", "col_38": "Data", "col_39": "Data", "col_40": "Data",
        "col_41": "Data", "col_42": "Data", "col_43": "Data", "col_44": "Data", "col_45": "Data",
        "col_46": "Data", "col_47": "Data", "col_48": "Data", "col_49": "Data", "col_50": "Data",
        "col_51": "Data", "col_52": "Data", "col_53": "Data", "col_54": "Data", "col_55": "Data",
        "col_56": "Data", "col_57": "Data", "col_58": "Data", "col_59": "Data", "col_60": "Data",
        "col_61": "Data", "col_62": "Data", "col_63": "Data", "col_64": "Data", "col_65": "Data",
        "col_66": "Data", "col_67": "Data", "col_68": "Data", "col_69": "Data", "col_70": "Data",
        "col_71": "Data", "col_72": "Data", "col_73": "Data", "col_74": "Data", "col_75": "Data",
        "col_76": "Data", "col_77": "Data", "col_78": "Data", "col_79": "Data", "col_80": "Data",
        "col_81": "Data", "col_82": "Data", "col_83": "Data", "col_84": "Data", "col_85": "Data",
        "col_86": "Data", "col_87": "Data", "col_88": "Data", "col_89": "Data", "col_90": "Data",
        "col_91": "Data", "col_92": "Data", "col_93": "Data", "col_94": "Data", "col_95": "Data",
        "col_96": "Data", "col_97": "Data", "col_98": "Data", "col_99": "Data", "col_100": "Data",
        "col_101": "Data", "col_102": "Data", "col_103": "Data", "col_104": "Data", "col_105": "Data",
        "col_106": "Data", "col_107": "Data", "col_108": "Data", "col_109": "Data", "col_110": "Data",
    }
    return report_metadata