#!/usr/bin/env python3
"""
전체 실험 결과를 하나의 표로 정리하는 스크립트
Usage: python collect_results.py
"""

import pandas as pd
import os
from pathlib import Path

def collect_all_results():
    """모든 실험 결과를 수집하여 통합 표 생성"""
    
    results_dir = Path("results")
    datasets = ["aras", "casas", "doore", "openpack", "opportunity"]
    baselines = ["calimera", "earliest", "stopandhop"]
    
    all_results = []
    
    for dataset in datasets:
        for baseline in baselines:
            summary_file = results_dir / dataset / f"{baseline}_kfold_summary.csv"
            
            if not summary_file.exists():
                print(f"⚠️  파일 없음: {summary_file}")
                continue
            
            # CSV 읽기
            df = pd.read_csv(summary_file)
            
            # 데이터 변환: metric을 컬럼으로
            result = {"dataset": dataset, "baseline": baseline}
            for _, row in df.iterrows():
                metric = row['metric']
                mean = row['mean']
                std = row['std']
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
            
            all_results.append(result)
    
    # DataFrame 생성
    results_df = pd.DataFrame(all_results)
    
    # 정렬
    results_df = results_df.sort_values(['dataset', 'baseline']).reset_index(drop=True)
    
    # 저장
    output_file = "results/all_experiments_summary.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ 통합 결과 저장: {output_file}")
    
    return results_df

def create_comparison_table():
    """비교표 생성 (보기 좋은 형식)"""
    
    results_dir = Path("results")
    datasets = ["aras", "casas", "doore", "openpack", "opportunity"]
    baselines = ["calimera", "earliest", "stopandhop"]
    
    # 각 baseline별로 데이터셋 행 구성
    comparison_data = []
    
    for dataset in datasets:
        row = {"Dataset": dataset}
        
        for baseline in baselines:
            summary_file = results_dir / dataset / f"{baseline}_kfold_summary.csv"
            
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                
                # accuracy와 f_e 추출
                acc_row = df[df['metric'] == 'accuracy']
                fe_row = df[df['metric'] == 'f_e']
                
                if not acc_row.empty and not fe_row.empty:
                    acc_mean = acc_row['mean'].values[0]
                    acc_std = acc_row['std'].values[0]
                    fe_mean = fe_row['mean'].values[0]
                    fe_std = fe_row['std'].values[0]
                    
                    row[f"{baseline}_acc"] = f"{acc_mean:.4f}±{acc_std:.4f}"
                    row[f"{baseline}_fe"] = f"{fe_mean:.4f}±{fe_std:.4f}"
                else:
                    row[f"{baseline}_acc"] = "N/A"
                    row[f"{baseline}_fe"] = "N/A"
            else:
                row[f"{baseline}_acc"] = "N/A"
                row[f"{baseline}_fe"] = "N/A"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 저장
    output_file = "results/comparison_table.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"✅ 비교표 저장: {output_file}")
    
    # 콘솔에 출력
    print("\n" + "="*80)
    print("전체 실험 결과 비교")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    return comparison_df

if __name__ == "__main__":
    print("🔍 실험 결과 수집 중...\n")
    
    # 통합 결과
    all_results = collect_all_results()
    
    # 비교표
    print()
    comparison = create_comparison_table()
    
    print("\n✅ 완료!")
