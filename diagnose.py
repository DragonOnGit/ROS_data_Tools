# -*- coding: utf-8 -*-
"""
Bag文件诊断工具 (增强版)
- 支持动态消息类型注册
- 提供详细的解析成功率统计
"""

import os
import sys


def show_pose_fields(msg):
    """显示消息中的位姿相关字段"""
    try:
        pose_paths = [
            ('msg.pose.pose', lambda m: m.pose.pose),
            ('msg.pose', lambda m: m.pose),
            ('msg.poses[0].pose', lambda m: m.poses[0].pose),
        ]
        
        for path, accessor in pose_paths:
            try:
                pose = accessor(msg)
                if hasattr(pose, 'position') and hasattr(pose, 'orientation'):
                    p = pose.position
                    o = pose.orientation
                    print(f"     Pose [{path}]:")
                    print(f"       position=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
                    print(f"       orientation=({o.w:.4f}, {o.x:.4f}, {o.y:.4f}, {o.z:.4f})")
                    return True
            except:
                pass
        
        attrs = [a for a in dir(msg) if not a.startswith('_')]
        print(f"     Available attributes: {attrs[:20]}")
        return False
        
    except Exception as e:
        print(f"     Error analyzing pose fields: {e}")
        return False


def create_enhanced_typestore(bag_path):
    """创建包含bag文件中所有消息类型的增强typestore
    
    Args:
        bag_path: bag文件路径
        
    Returns:
        增强后的typestore对象
    """
    from rosbags.typesys import get_typestore, Stores, get_types_from_msg
    from rosbags.rosbag1.reader import Reader
    
    # 加载基础typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    # 动态注册bag文件中的所有消息类型
    registered = 0
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            if connection.msgdef and connection.msgtype:
                try:
                    # 检查是否已存在
                    typestore.get_msgdef(connection.msgtype)
                except Exception:
                    try:
                        types = get_types_from_msg(connection.msgdef, connection.msgtype)
                        typestore.register(types)
                        registered += 1
                    except Exception:
                        pass
    
    return typestore, registered


def diagnose_bag_file(bag_path: str, target_topic: str = None):
    """诊断bag文件的解析情况（使用增强的typestore）"""
    
    print("=" * 70)
    print("ROS Bag File Diagnostic Tool (Enhanced)")
    print("=" * 70)
    
    if not os.path.exists(bag_path):
        print(f"[ERROR] File not found: {bag_path}")
        return
    
    from rosbags.rosbag1.reader import Reader
    from rosbags.serde import deserialize_cdr
    
    print("\n[INFO] Checking rosbags library...")
    
    # 创建增强的typestore
    print("[INFO] Creating enhanced typestore with dynamic message registration...")
    typestore, reg_count = create_enhanced_typestore(bag_path)
    print(f"[OK] Base typestore loaded + {reg_count} custom types registered")
    
    # 读取bag文件
    print(f"\n[INFO] Analyzing file: {bag_path}")
    
    with Reader(bag_path) as reader:
        connections = reader.connections
        print(f"\n[INFO] Total connections: {len(connections)}")
        
        all_success = True
        
        for i, connection in enumerate(connections):
            if target_topic and connection.topic != target_topic:
                continue
                
            print(f"\n{'='*60}")
            print(f"Topic #{i+1}: {connection.topic}")
            print(f"  Message Type: {connection.msgtype}")
            
            msg_count = 0
            success_count = 0
            error_count = 0
            error_types = {}
            
            print(f"\n  Testing message parsing...")
            
            for conn, timestamp, rawdata in reader.messages():
                if conn.topic == connection.topic:
                    msg_count += 1
                    
                    try:
                        msg = deserialize_cdr(rawdata, connection.msgtype, typestore)
                        success_count += 1
                        
                        if success_count <= 3:
                            print(f"  [OK] Message #{msg_count} parsed successfully")
                            show_pose_fields(msg)
                            
                    except UnicodeDecodeError as ude:
                        error_count += 1
                        key = "UnicodeDecodeError"
                        error_types[key] = error_types.get(key, 0) + 1
                        if error_count <= 2:
                            print(f"  [WARN] Message #{msg_count}: {str(ude)[:60]}...")
                            
                    except TypeError as te:
                        error_count += 1
                        key = f"TypeError"
                        error_types[key] = error_types.get(key, 0) + 1
                        if error_count <= 2:
                            print(f"  [ERROR] Message #{msg_count}: {str(te)[:60]}")
                            
                    except Exception as e:
                        error_count += 1
                        key = f"{type(e).__name__}"
                        error_types[key] = error_types.get(key, 0) + 1
                        if error_count <= 2:
                            print(f"  [ERROR] Message #{msg_count}: {type(e).__name__}: {str(e)[:60]}")
                
                # 抽样检查前50条消息以加快速度
                if msg_count >= 50:
                    break
            
            print(f"\n  Statistics for '{connection.topic}':")
            print(f"    Sampled messages: {msg_count}")
            print(f"    Successfully parsed: {success_count}")
            print(f"    Failed/Skipped: {error_count}")
            
            rate = (success_count / msg_count * 100) if msg_count > 0 else 0
            print(f"    Success rate: {rate:.1f}%")
            
            if error_count > 0:
                print(f"\n    Error distribution:")
                for err_type, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
                    print(f"      - {err_type}: {count} times")
                
                if rate < 100:
                    all_success = False
                    
                if rate < 50:
                    print(f"\n    [WARNING] Low success rate! This topic may have issues.")
            else:
                print(f"\n    [PERFECT] All messages parsed successfully!")
        
        print(f"\n{'='*70}")
        
        if all_success:
            print("[RESULT] All topics can be parsed successfully!")
            print("The program should work correctly with this bag file.")
        else:
            print("[RESULT] Some topics have parsing issues.")
            print("The enhanced typestore has been created but some types may still fail.")
        
        print("=" * 70)


def main():
    print("\nUsage: python diagnose.py <bag_file_path> [topic_name]\n")
    
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]
    else:
        bag_file = input("Enter bag file path: ").strip()
    
    if bag_file and os.path.exists(bag_file):
        topic = sys.argv[2] if len(sys.argv) > 2 else None
        diagnose_bag_file(bag_file, topic)
    elif not bag_file:
        print("No file path provided")
    else:
        print(f"File not found: {bag_file}")


if __name__ == "__main__":
    main()
