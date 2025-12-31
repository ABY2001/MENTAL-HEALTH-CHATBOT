import { Routes } from '@angular/router';
import { MainLayout } from './core/layout/main-layout/main-layout';

export const routes: Routes = [
    {
        path:'login',
        loadChildren: ()=> import('./features/login/login.routing').then(m =>m.LoginRoutes)
        
    },
    {
      path: '',redirectTo: '/login',pathMatch: 'full'
    },
    
    {
        path: '',
        component: MainLayout,
        children:[
             {
                path: 'audio-record',
                loadComponent: ()=> import('./features/audio-record/audio-record').then(c =>c.AudioRecord)
            },
        ]
    }  
];
